// DiT (diffusion transformer) glue kernels — BF16 I/O, FP32 compute.
//
// BF16 twins of the norm + row-indexed modulation kernels in dit_f32.cu, for DiT bodies that run BF16
// activations (BF16's F32-matching exponent range is what lets a deep video DiT drop to 16-bit
// without the F16 overflow guards). Same convention as dit_f16.cu: the activation tensor is
// __nv_bfloat16, the per-channel modulation table stays FP32 — it is tiny ([modRows, dim], a few
// rows) and precision-sensitive. FP32 accumulation throughout.
//
// Build (no nvcc on this box — use the repo's nvrtc frontend, which needs the CUDA headers for
// cuda_bf16.h):
//   LD_LIBRARY_PATH=~/.local/lib/cuda13 src/HartsyInference.Cuda/Kernels/nvrtc_compile \
//     src/HartsyInference.Cuda/Kernels/dit/dit_bf16.cu src/HartsyInference.Cuda/Kernels/dit/dit_bf16.ptx \
//     compute_80 ~/.local/lib/cuda13/include
//   cp src/HartsyInference.Cuda/Kernels/dit/dit_bf16.ptx src/HartsyInference.Cuda/Ptx/

#include <cuda_bf16.h>

extern "C" {

// ── RMSNorm (BF16 I/O, F32 accumulate; weight stays F32) ───────────────────
// out[row,i] = in[row,i] * rsqrt(mean(in[row,:]^2) + eps) * weight[i].
// One block per row; also serves per-head QK-RMSNorm (rows = B*L*numHeads, normDim = headDim).
__global__ void dit_rmsnorm_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ weight,
    unsigned int normDim,
    unsigned int totalRows,
    float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;

    const __nv_bfloat16* inRow = input + (size_t)row * normDim;
    __nv_bfloat16* outRow = output + (size_t)row * normDim;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < normDim; i += blockDim.x)
    {
        float v = __bfloat162float(inRow[i]);
        partial += v * v;
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float invRms = rsqrtf(sdata[0] / (float)normDim + eps);
    for (unsigned int i = threadIdx.x; i < normDim; i += blockDim.x)
        outRow[i] = __float2bfloat16(__bfloat162float(inRow[i]) * invRms * weight[i]);
}

// ── Row-indexed gated residual (gather fused in) ───────────────────────────
// out[r,d] = residual[r,d] + gateTable[rowIndex[r], d] * value[r,d]
__global__ void dit_gated_residual_rowindexed_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gateTable,
    const int* __restrict__ rowIndex,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    size_t tIdx = (size_t)rowIndex[row] * dim + d;

    output[i] = __float2bfloat16(__bfloat162float(residual[i]) + gateTable[tIdx] * __bfloat162float(value[i]));
}

// ── Gated residual over the last dim (BF16 activations, F32 gate) ──────────
// out[b,s,d] = residual[b,s,d] + gate[b,d] * value[b,s,d]. gate broadcast over S.
__global__ void dit_gated_residual_lastdim_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    unsigned int seqLen,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    unsigned int b = (unsigned int)(row / seqLen);
    size_t pIdx = (size_t)b * dim + d;

    output[i] = __float2bfloat16(__bfloat162float(residual[i]) + gate[pIdx] * __bfloat162float(value[i]));
}

} // extern "C"
