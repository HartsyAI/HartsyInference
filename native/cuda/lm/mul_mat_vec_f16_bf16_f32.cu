// Fused BF16/F16 × F32 matrix-vector product for LLM decode (M small, M=1 hot path).
//
// The dense-float counterpart to mul_mat_vec_q*_f32: for checkpoints that ship 16-bit float weights
// (Orpheus, NeuTTS, and most audio LMs are BF16), cuBLAS GemmEx is inefficient at M=1 — this reads
// each weight row once with an F32 accumulate, one warp per output row, exactly like the quant GEMVs.
// Activations stay F32 (not downcast), so this is at least as accurate as the cuBLAS BF16 path.
//
// weight: [N, K] row-major (bf16/f16). input: [M, K] F32. output: [M, N] F32. bias: [N] F32 or null.
// Launch: blockDim = (32, WARPS_PER_BLOCK); grid = (ceil(N / WARPS_PER_BLOCK), M). One warp per row.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32

extern "C" __global__ void mul_mat_vec_bf16_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;                          // 0..31
    const int n = blockIdx.x * blockDim.y + threadIdx.y;   // output row
    const int m = blockIdx.y;                              // batch row
    if (n >= N || m >= M) return;

    const __nv_bfloat16* wrow = weight + (size_t)n * K;
    const float* xrow = input + (size_t)m * K;

    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE) {
        acc += __bfloat162float(wrow[k]) * xrow[k];
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

extern "C" __global__ void mul_mat_vec_f16_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const __half* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;
    const int n = blockIdx.x * blockDim.y + threadIdx.y;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const __half* wrow = weight + (size_t)n * K;
    const float* xrow = input + (size_t)m * K;

    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE) {
        acc += __half2float(wrow[k]) * xrow[k];
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
