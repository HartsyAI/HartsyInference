// Fused Q3_K × F32 matrix-vector product for LLM decode (M small): the warp-per-row scheme of
// mul_mat_vec_q4k_f32.cu with dequant_q3_k_to_f16.cu's per-element decode inlined. Each lane owns 8
// contiguous elements of every 256-wide super-block, which share one (half, shift group, 16-run), so a lane
// reads one 6-bit scale, one 8-byte run of quants and one 8-byte run of the high-bit mask per super-block.
//
// Q3_K layout (110 bytes): [32 hmask][64 qs: low 2 bits][12 scales: 6-bit signed, packed][d].
// x = d · scale · (q − (mask bit set ? 0 : 4)) — the INVERTED high bit, exactly as the dequant kernel reads it.
//
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define SUPER_ELEMS 256
#define SUPER_BYTES 110
#define WARP_SIZE 32

// Byte-assembled little-endian word: a K-quant super-block is 84/110 bytes, so no wider load is aligned.
__device__ __forceinline__ unsigned int load_u32(const unsigned char* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

// Canonical ggml 6-bit signed scale unpack (dequantize_row_q3_K): entry s takes its low nibble from scales[s]
// (s < 8) or the high nibble of scales[s - 8], and its two high bits from scales[8 + s % 4] at bit 2·(s / 4).
__device__ __forceinline__ int unpack_q3k_scale(const unsigned char* packed, int index)
{
    const int low = index < 8 ? (packed[index] & 0x0F) : (packed[index - 8] >> 4);
    const int high = (packed[8 + (index & 3)] >> (2 * (index >> 2))) & 0x03;
    return (low | (high << 4)) - 32;
}

extern "C" __global__ void mul_mat_vec_q3k_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int nsb = K / SUPER_ELEMS;
    const unsigned char* wrow = weight + (size_t)n * nsb * SUPER_BYTES;
    const float* xrow = input + (size_t)m * K;

    const int e0 = lane * 8;
    const int h = e0 >> 7;
    const int j = (e0 & 127) >> 5;
    const int half16 = (e0 & 31) >> 4;
    const int l0 = e0 & 15;
    const int scaleIdx = 8 * h + 2 * j + half16;
    const int qsOff = 32 * h + 16 * half16 + l0;
    const int hmOff = 16 * half16 + l0;
    const int shift = 2 * j;
    const unsigned int maskBit = 1u << (4 * h + j);

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* block = wrow + (size_t)sb * SUPER_BYTES;
        const float d = __half2float(*(const __half*)(block + 108));
        const float scale = d * (float)unpack_q3k_scale(block + 96, scaleIdx);

        const unsigned char* q = block + 32 + qsOff;
        const unsigned char* hm = block + hmOff;
        const unsigned int lo = load_u32(q), hi = load_u32(q + 4), mlo = load_u32(hm), mhi = load_u32(hm + 4);
        const float* xb = xrow + sb * SUPER_ELEMS + e0;
        const float4 xa = *reinterpret_cast<const float4*>(xb);
        const float4 xb2 = *reinterpret_cast<const float4*>(xb + 4);
        #define Q3W(word, mword, b) \
            (scale * (float)((int)(((word) >> (shift + 8 * (b))) & 3u) - ((((mword) >> (8 * (b))) & maskBit) ? 0 : 4)))
        const float w0 = Q3W(lo, mlo, 0), w1 = Q3W(lo, mlo, 1), w2 = Q3W(lo, mlo, 2), w3 = Q3W(lo, mlo, 3);
        const float w4 = Q3W(hi, mhi, 0), w5 = Q3W(hi, mhi, 1), w6 = Q3W(hi, mhi, 2), w7 = Q3W(hi, mhi, 3);
        #undef Q3W
        acc += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
             + w4 * xb2.x + w5 * xb2.y + w6 * xb2.z + w7 * xb2.w;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        if (bias != nullptr) acc += bias[n];
        output[(size_t)m * N + n] = acc;
    }
}
