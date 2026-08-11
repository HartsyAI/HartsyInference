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

// ── Row-indexed broadcast affine (gather fused in) ─────────────────────────
// out[r,d] = in[r,d] * (1 + scaleTable[rowIndex[r], d]) + (shiftTable ? shiftTable[rowIndex[r], d] : 0)
// Indexing the small table through rowIndex is what removes the materialized [seq, dim] gathers;
// the `1 +` is folded in too, removing the caller's AddScalar over the table.
__global__ void dit_affine_broadcast_rowindexed_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ scaleTable,
    const float* __restrict__ shiftTable,
    const int* __restrict__ rowIndex,
    unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    size_t tIdx = (size_t)rowIndex[row] * dim + d;

    float v = __bfloat162float(input[i]) * (1.0f + scaleTable[tIdx]);
    if (shiftTable != 0) v += shiftTable[tIdx];
    output[i] = __float2bfloat16(v);
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

// ── Broadcast affine over the last dim (BF16 activation, F32 scale/shift) ───
// out[b,s,d] = in[b,s,d] * scale[b,d] + (shift ? shift[b,d] : 0). BF16 twin of the F32/F16 kernels,
// for the call sites that modulate one broadcast row (a DiT final layer's per-segment head) — an
// index tensor would be pure overhead there.
__global__ void dit_affine_broadcast_lastdim_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
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

    float v = __bfloat162float(input[i]) * scale[pIdx];
    if (shift != 0) v += shift[pIdx];
    output[i] = __float2bfloat16(v);
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

// ── GEGLU (BF16 I/O, FP32 compute) ──────────────────────────────────────────
// input is [..., 2*innerDim] with [value | gate] in each logical row. The split must remain on
// the last dimension; treating the flat allocation as two halves is only correct for a single row.
__global__ void dit_geglu_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    unsigned int innerDim,
    unsigned int outputElements)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outputElements) return;

    unsigned int outer = i / innerDim;
    unsigned int d = i % innerDim;
    size_t inputIndex = (size_t)outer * (2u * innerDim) + d;
    float value = __bfloat162float(input[inputIndex]);
    float gate = __bfloat162float(input[inputIndex + innerDim]);
    float inner = 0.7978845608028654f * (gate + 0.044715f * gate * gate * gate);
    float sigmoid = __fdividef(1.0f, 1.0f + __expf(-2.0f * inner));
    float gelu = 0.5f * gate * (1.0f + (2.0f * sigmoid - 1.0f));
    output[i] = __float2bfloat16(value * gelu);
}

// ── Gated-FFN activation epilogue (BF16 I/O, F32 compute) ──────────────────
// comb[r*ff+i] = act(gateUp[r, i]) * gateUp[r, ff+i] over the CONCATENATED [gate | up] projection.
// BF16 twin of lm_glu_act_f32 — same fast-math formulas, same flat-index-to-(row, i) decomposition
// (the split is on the LAST dim, not the flat midpoint). act 0 = SiLU, 1 = GELU-tanh.
__global__ void dit_glu_act_bf16(
    __nv_bfloat16* __restrict__ comb,
    const __nv_bfloat16* __restrict__ gateUp,
    unsigned int rows,
    unsigned int ff,
    int act)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)rows * ff;
    if (gid >= total) return;
    unsigned int r = (unsigned int)(gid / ff);
    unsigned int i = (unsigned int)(gid % ff);
    unsigned long long rowBase = (unsigned long long)r * 2u * ff;
    const float g = __bfloat162float(gateUp[rowBase + i]);
    const float u = __bfloat162float(gateUp[rowBase + ff + i]);
    float a;
    if (act == 0) {
        a = g * __fdividef(1.0f, 1.0f + __expf(-g));                    // SiLU
    } else {
        const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        const float s = __fdividef(1.0f, 1.0f + __expf(-2.0f * inner)); // tanh(x) = 2·sigmoid(2x) − 1
        a = 0.5f * g * (1.0f + (2.0f * s - 1.0f));                      // GELU-tanh
    }
    comb[gid] = __float2bfloat16(a * u);
}

} // extern "C"
