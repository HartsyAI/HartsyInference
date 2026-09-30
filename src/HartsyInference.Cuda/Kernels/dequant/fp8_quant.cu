// Per-tensor dynamic FP8 activation quantization for the native cuBLASLt FP8 GEMM path (Ada+).
// Three kernels: two-pass absmax reduction, then a fused scale-and-quantize F32 -> e4m3fn.
// The dequant scale (amax/448) is written to DEVICE memory and consumed by cublasLtMatmul via
// CUBLASLT_MATMUL_DESC_B_SCALE_POINTER — the whole quantize+GEMM chain stays async (no host sync).
// The e4m3 conversion (block_scale.cuh) is hand-rolled bit math (no sm_89-only cvt instructions) so this PTX
// still JITs on Ampere for the unit tests; the GEMM itself is gated on SM 8.9+ elsewhere.
#include "block_scale.cuh"

#define REDUCE_THREADS 256u

// Pass 1: grid-strided per-block |max| -> blockMax[blockIdx].
extern "C" __global__ void absmax_f32(
    const float* __restrict__ x, float* __restrict__ blockMax, unsigned int n)
{
    __shared__ float sm[REDUCE_THREADS];
    unsigned int tid = threadIdx.x;
    float m = 0.0f;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + tid; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
    {
        float v = fabsf(x[i]);
        if (v > m) m = v;
    }
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = sm[0];
}

// Pass 2 (single block): blockMax[0..numBlocks) -> scale[0] = amax/448 (the e4m3 DEQUANT scale).
// amax==0 (all-zero tensor) writes scale 1.0 so the GEMM stays a well-defined no-op.
extern "C" __global__ void absmax_finalize_scale(
    const float* __restrict__ blockMax, unsigned int numBlocks, float* __restrict__ scale)
{
    __shared__ float sm[REDUCE_THREADS];
    unsigned int tid = threadIdx.x;
    float m = 0.0f;
    for (unsigned int i = tid; i < numBlocks; i += blockDim.x)
    {
        float v = blockMax[i];
        if (v > m) m = v;
    }
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) scale[0] = sm[0] > 0.0f ? sm[0] / 448.0f : 1.0f;
}

// out[i] = e4m3(x[i] * 448/amax), reading the dequant scale written by absmax_finalize_scale.
extern "C" __global__ void quant_f32_e4m3(
    const float* __restrict__ x, unsigned char* __restrict__ out,
    const float* __restrict__ scale, unsigned int n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float rs = 1.0f / scale[0];
    out[i] = f32_to_e4m3(x[i] * rs);
}

// ── F16-activation twins (the Krea2 F16-activation path) ───────────────────────────────────────
// Same two-pass absmax + fused scale-and-quantize, reading __half activations (converted to F32 in
// registers — the reduction/quant math is unchanged). Keeps the native fp8 GEMM path (weights packed,
// activation quantized per-tensor) available to F16 activations; without these, F16 activations fall
// off the native path onto the per-call fp8->F16 weight recast (the Axis-B pathology). __half is read
// via raw bit reinterpretation (h16_to_f32 in block_scale.cuh; no cuda_fp16.h dependency).
extern "C" __global__ void absmax_f16(
    const unsigned short* __restrict__ x, float* __restrict__ blockMax, unsigned int n)
{
    __shared__ float sm[REDUCE_THREADS];
    unsigned int tid = threadIdx.x;
    float m = 0.0f;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + tid; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
    {
        float v = fabsf(h16_to_f32(x[i]));
        if (v > m) m = v;
    }
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s && sm[tid + s] > sm[tid]) sm[tid] = sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) blockMax[blockIdx.x] = sm[0];
}

extern "C" __global__ void quant_f16_e4m3(
    const unsigned short* __restrict__ x, unsigned char* __restrict__ out,
    const float* __restrict__ scale, unsigned int n)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float rs = 1.0f / scale[0];
    out[i] = f32_to_e4m3(h16_to_f32(x[i]) * rs);
}
