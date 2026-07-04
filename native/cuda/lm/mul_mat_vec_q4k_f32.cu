// Fused Q4_K × F32 matrix-vector product for LLM decode (M small).
//
// Computes  output[m, n] = sum_k input[m, k] * dequant(weight[n, k]) [ + bias[n] ]
// reading the Q4_K weight bytes ONCE and dequantizing inline (no F16 materialization).
// This is the decode-path replacement for "dequant whole weight to F16 (extra HBM pass)
// then cuBLAS GEMM at M=1" — it moves ~1/4 the weight bytes and skips the intermediate.
//
// Layout:
//   output : [M, N] F32, row-major
//   input  : [M, K] F32, row-major
//   weight : [N, K] Q4_K, row-major. Each row = K/256 super-blocks of 144 bytes.
//            (Q4_K quantizes along K, so K is always a multiple of 256 and every row
//             starts on a super-block boundary.)
//   bias   : [N] F32 or nullptr
//
// One WARP per output row (32 lanes), WARPS_PER_BLOCK rows per block. Each lane owns 8
// contiguous elements of every 256-wide super-block (32 lanes × 8 = 256), accumulates its
// partial dot product across the row's super-blocks in F32, then a warp-shuffle reduction
// yields the result — no shared memory, no __syncthreads. F32 accumulate (more accurate
// than the old BF16-operand cuBLAS path).
//
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 144
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

extern "C" __global__ void mul_mat_vec_q4k_f32(
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
    const int subByteBase = (j >> 1) * SUB_ELEMS + base_i;  // qs offset for this lane's 8 nibbles
    const int xBase = j * SUB_ELEMS + base_i;               // element offset within a super-block

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const float d = __half2float(*(const __half*)block);
        const float dmin = __half2float(*(const __half*)(block + 2));
        const unsigned char* scales = block + 4;
        const unsigned char* qs = block + 16;

        unsigned char sc, mm;
        get_scale_min_k4(j, scales, &sc, &mm);
        const float subScale = d * (float)sc;
        const float subMin = dmin * (float)mm;

        const float* xb = xrow + sb * SUPER_ELEMS + xBase;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int q = (qs[subByteBase + t] >> nibbleShift) & 0x0F;
            const float w = subScale * (float)q - subMin;
            acc += w * xb[t];
        }
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
