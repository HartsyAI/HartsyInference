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

// Small-M fast path: one warp handles a weight row for ALL M batch rows at once, so each 16-bit
// weight element is loaded from DRAM exactly once regardless of M. With grid.y = M (the old layout)
// batch row 1's blocks ran after the whole matrix had streamed for row 0, so L2 held only a fraction
// of the rows and an M=2 call cost ~1.27× an M=1 call instead of ~1.0× (measured on the HeartMuLa
// CFG-batched decode, 4090, 2026-07-25). Per-(m,row) accumulation order is unchanged → bit-identical.
#define MMV_MAX_M 4

extern "C" __global__ void mul_mat_vec_bf16_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,     // may be nullptr
    int N, int K, int M)
{
    const int lane = threadIdx.x;                          // 0..31
    const int n = blockIdx.x * blockDim.y + threadIdx.y;   // output row
    if (n >= N) return;

    const __nv_bfloat16* wrow = weight + (size_t)n * K;

    if (M >= 2 && M <= MMV_MAX_M && blockIdx.y == 0) {
        float acc[MMV_MAX_M];
        #pragma unroll
        for (int m = 0; m < MMV_MAX_M; ++m) acc[m] = 0.0f;
        for (int k = lane; k < K; k += WARP_SIZE) {
            const float w = __bfloat162float(wrow[k]);
            #pragma unroll
            for (int m = 0; m < MMV_MAX_M; ++m) {
                if (m < M) acc[m] += w * input[(size_t)m * K + k];
            }
        }
        #pragma unroll
        for (int m = 0; m < MMV_MAX_M; ++m) {
            float a = acc[m];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xffffffffu, a, offset);
            }
            if (lane == 0 && m < M) {
                if (bias != nullptr) a += bias[n];
                output[(size_t)m * N + n] = a;
            }
        }
        return;
    }

    const int m1 = blockIdx.y;                             // large-M fallback: original one-(row,m)-per-warp layout
    if (m1 >= M) return;
    const float* xrow = input + (size_t)m1 * K;

    float accS = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE) {
        accS += __bfloat162float(wrow[k]) * xrow[k];
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        accS += __shfl_down_sync(0xffffffffu, accS, offset);
    }
    if (lane == 0) {
        if (bias != nullptr) accS += bias[n];
        output[(size_t)m1 * N + n] = accS;
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
    if (n >= N) return;

    const __half* wrow = weight + (size_t)n * K;

    if (M >= 2 && M <= MMV_MAX_M && blockIdx.y == 0) {   // same small-M row-reuse path as the bf16 kernel above
        float acc[MMV_MAX_M];
        #pragma unroll
        for (int m = 0; m < MMV_MAX_M; ++m) acc[m] = 0.0f;
        for (int k = lane; k < K; k += WARP_SIZE) {
            const float w = __half2float(wrow[k]);
            #pragma unroll
            for (int m = 0; m < MMV_MAX_M; ++m) {
                if (m < M) acc[m] += w * input[(size_t)m * K + k];
            }
        }
        #pragma unroll
        for (int m = 0; m < MMV_MAX_M; ++m) {
            float a = acc[m];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xffffffffu, a, offset);
            }
            if (lane == 0 && m < M) {
                if (bias != nullptr) a += bias[n];
                output[(size_t)m * N + n] = a;
            }
        }
        return;
    }

    const int m1 = blockIdx.y;
    if (m1 >= M) return;
    const float* xrow = input + (size_t)m1 * K;

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
        output[(size_t)m1 * N + n] = acc;
    }
}
