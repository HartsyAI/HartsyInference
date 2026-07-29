// Fused Q8_0 × F32 matrix-vector product for LLM decode (M small).
//
// Same design as mul_mat_vec_q4k_f32: read the Q8_0 weight bytes ONCE, dequant inline,
// F32 accumulate, one warp per output row.
//
// Q8_0 block: 32 elements / 34 bytes: [2 B fp16 d] [32 B int8 quants]. x[i] = d * q[i].
// One warp (32 lanes) per output row; lane `l` owns element `l` of every 32-wide block.
//
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M).

#include <cuda_fp16.h>

#define BLK_ELEMS 32
#define BLK_BYTES 34
#define WARP_SIZE 32

extern "C" __global__ void mul_mat_vec_q8_0_f32(
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

    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        const unsigned char* block = wrow + (size_t)b * BLK_BYTES;
        const float d = __half2float(*(const __half*)block);
        const signed char* qs = (const signed char*)(block + 2);
        acc += d * (float)qs[lane] * xrow[b * BLK_ELEMS + lane];
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
