// Fused Q5_K × F32 matrix-vector product for LLM decode (M small).
//
// Same geometry as mul_mat_vec_q4k_f32 (K-quant super-block layout, one warp per output row,
// each lane owns 8 contiguous elements of a sub-block), extended with Q5_K's extra high-bit
// plane. Without this kernel Q5_K tensors miss every fused GEMV path and fall back to the
// ~10-20× slower dequant-to-F16-then-cuBLAS route.
//
// Layout:
//   output : [M, N] F32, row-major
//   input  : [M, K] F32, row-major
//   weight : [N, K] Q5_K, row-major. Each row = K/256 super-blocks of 176 bytes:
//            [2B d][2B dmin][12B packed 6-bit scales+mins][32B high bits][128B low nibbles]
//   bias   : [N] F32 or nullptr
//
// Reconstruction: x = d*sc_j*q - dmin*m_j, q = low | (high << 4) ∈ [0,31]
// (matches dequant_q5_k_to_f16 exactly — ground truth for the bit layout).
//
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 176
#define SUB_ELEMS 32
#define WARP_SIZE 32

// Canonical ggml get_scale_min_k4 — 6-bit scale + 6-bit min for sub-block j (0..7).
__device__ __forceinline__ void get_scale_min_k4(
    int j, const unsigned char* q, unsigned char* d, unsigned char* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

extern "C" __global__ void mul_mat_vec_q5k_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;                          // 0..31
    const int n = blockIdx.x * blockDim.y + threadIdx.y;   // output row
    const int m = blockIdx.y;                              // batch row
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;                       // super-blocks per row
    const unsigned char* wrow = weight + (size_t)n * nsb * SUPER_BYTES;
    const float* xrow = input + (size_t)m * K;

    // This lane owns sub-block j and the 8 elements [base_i, base_i+8) within it.
    const int j = lane >> 2;               // lane/4  -> 0..7
    const int base_i = (lane & 3) << 3;    // (lane%4)*8 -> {0,8,16,24}
    const int nibbleShift = (j & 1) ? 4 : 0;
    const int subByteBase = (j >> 1) * SUB_ELEMS + base_i;   // low-nibble byte offset for this lane's 8 elements
    const int xBase = j * SUB_ELEMS + base_i;                // element offset within a super-block

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const float d = __half2float(*(const __half*)block);
        const float dmin = __half2float(*(const __half*)(block + 2));
        const unsigned char* scales = block + 4;
        const unsigned char* highBits = block + 16;   // 32 bytes, bit j of byte i = high bit of element i, sub-block j
        const unsigned char* qs = block + 48;         // 128 bytes, low nibbles (Q4_K-style packing)

        unsigned char sc, mm;
        get_scale_min_k4(j, scales, &sc, &mm);
        const float subScale = d * (float)sc;
        const float subMin = dmin * (float)mm;

        // Vectorized loads: both the 8 low-nibble bytes and the 8 high-bit bytes for this lane's
        // elements are 8-byte aligned (SUPER_BYTES=176 and all offsets are multiples of 8),
        // so a uint2 load replaces 16 scalar byte reads.
        const float* xb = xrow + sb * SUPER_ELEMS + xBase;
        const uint2 qpack = *reinterpret_cast<const uint2*>(qs + subByteBase);
        const uint2 hpack = *reinterpret_cast<const uint2*>(highBits + base_i);
        const float4 xa = *reinterpret_cast<const float4*>(xb);
        const float4 xb2 = *reinterpret_cast<const float4*>(xb + 4);
        const unsigned int lo = qpack.x, hi = qpack.y;
        const unsigned int hlo = hpack.x, hhi = hpack.y;

        // Byte k of hlo/hhi holds the high bits (one per sub-block, bit-indexed by j) for element
        // base_i+k (k=0..3 from hlo, k=4..7 from hhi) — bit position (8*k + j) within the word.
        const int hb0 = (int)((hlo >> ( 0 + j)) & 0x1u);
        const int hb1 = (int)((hlo >> ( 8 + j)) & 0x1u);
        const int hb2 = (int)((hlo >> (16 + j)) & 0x1u);
        const int hb3 = (int)((hlo >> (24 + j)) & 0x1u);
        const int hb4 = (int)((hhi >> ( 0 + j)) & 0x1u);
        const int hb5 = (int)((hhi >> ( 8 + j)) & 0x1u);
        const int hb6 = (int)((hhi >> (16 + j)) & 0x1u);
        const int hb7 = (int)((hhi >> (24 + j)) & 0x1u);

        const int q0 = (int)(((lo      ) >> nibbleShift) & 0xF) | (hb0 << 4);
        const int q1 = (int)(((lo >>  8) >> nibbleShift) & 0xF) | (hb1 << 4);
        const int q2 = (int)(((lo >> 16) >> nibbleShift) & 0xF) | (hb2 << 4);
        const int q3 = (int)(((lo >> 24) >> nibbleShift) & 0xF) | (hb3 << 4);
        const int q4 = (int)(((hi      ) >> nibbleShift) & 0xF) | (hb4 << 4);
        const int q5 = (int)(((hi >>  8) >> nibbleShift) & 0xF) | (hb5 << 4);
        const int q6 = (int)(((hi >> 16) >> nibbleShift) & 0xF) | (hb6 << 4);
        const int q7 = (int)(((hi >> 24) >> nibbleShift) & 0xF) | (hb7 << 4);

        const float w0 = subScale * (float)q0 - subMin;
        const float w1 = subScale * (float)q1 - subMin;
        const float w2 = subScale * (float)q2 - subMin;
        const float w3 = subScale * (float)q3 - subMin;
        const float w4 = subScale * (float)q4 - subMin;
        const float w5 = subScale * (float)q5 - subMin;
        const float w6 = subScale * (float)q6 - subMin;
        const float w7 = subScale * (float)q7 - subMin;
        acc += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
             + w4 * xb2.x + w5 * xb2.y + w6 * xb2.z + w7 * xb2.w;
    }

    // Warp-shuffle reduction across the 32 lanes.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}
