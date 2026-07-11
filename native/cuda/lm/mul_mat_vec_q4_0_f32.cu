// Fused Q4_0 × F32 matrix-vector product for LLM decode (M small).
//
// Same design as mul_mat_vec_q5_0_f32: read the Q4_0 weight bytes ONCE, dequant inline,
// F32 accumulate, one warp per output row. Q4_0 is a common legacy/baseline GGUF quant that,
// without this kernel, misses every fused GEMV path and falls back to the ~10-20× slower
// dequant-to-F16-then-cuBLAS route.
//
// Q4_0 block: 32 elements / 18 bytes: [2B fp16 scale d][16B packed nibbles qs].
// x[j]    = d * ((qs[j] & 0x0F) - 8)   for j in [0,16)
// x[j+16] = d * ((qs[j] >>   4) - 8)   for j in [0,16)
// (matches dequant_q4_0_to_f16 exactly — ground truth for the bit layout).
// One warp (32 lanes) per output row; lane l owns element l.
//
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define BLK_ELEMS 32
#define BLK_BYTES 18
#define HALF_BLOCK 16
#define WARP_SIZE 32

extern "C" __global__ void mul_mat_vec_q4_0_f32(
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

    const int nblk = K / BLK_ELEMS;
    const unsigned char* wrow = weight + (size_t)n * nblk * BLK_BYTES;
    const float* xrow = input + (size_t)m * K;

    const int subIdx = lane < HALF_BLOCK ? lane : lane - HALF_BLOCK;   // byte index within the 16-byte nibble pack
    const bool highHalf = lane >= HALF_BLOCK;

    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        const unsigned char* block = wrow + (size_t)b * BLK_BYTES;
        const float scale = __half2float(*(const __half*)block);
        const unsigned char qByte = block[2 + subIdx];
        const int nibble = highHalf ? ((qByte >> 4) & 0xF) : (qByte & 0xF);
        acc += scale * (float)(nibble - 8) * xrow[b * BLK_ELEMS + lane];
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
