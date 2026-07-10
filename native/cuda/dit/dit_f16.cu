// DiT (diffusion transformer) glue kernels — FP16 I/O, FP32 compute.
//
// Half-precision twins of the hot per-block glue ops in dit_f32.cu / wan_rope.cu / lm_f32.cu /
// audio_activations_f32.cu, added so the Krea2 (and any) DiT can run F16 activations and halve the
// HBM traffic of the bandwidth-bound elementwise/norm/rope/gate kernels. The primary activation
// tensor is __half; per-channel parameter vectors (rms weight, modulation scale/shift/gate, RoPE
// cos/sin) stay FP32 — they are tiny and precision-sensitive, and the model already allocates them
// FP32. FP32 accumulation throughout (read __half → compute float → write __half), matching the
// KERNEL.md standard.
//
// Build (no nvcc on this box — use the repo's nvrtc frontend):
//   LD_LIBRARY_PATH=~/.local/lib/cuda13 native/cuda/nvrtc_compile \
//     native/cuda/dit/dit_f16.cu native/cuda/dit/dit_f16.ptx compute_80
//   cp native/cuda/dit/dit_f16.ptx src/HartsyInference.Cuda/Ptx/

#include <cuda_fp16.h>

extern "C" {

// ── RMSNorm (F16 I/O, F32 accumulate; weight stays F32) ─────────────────────
// out[row,i] = in[row,i] * rsqrt(mean(in[row,:]^2) + eps) * weight[i].
// One block per row; also serves per-head QK-RMSNorm (rows = B*L*numHeads, normDim = headDim).
__global__ void dit_rmsnorm_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const float* __restrict__ weight,
    unsigned int normDim,
    unsigned int totalRows,
    float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;

    const __half* inRow = input + (size_t)row * normDim;
    __half* outRow = output + (size_t)row * normDim;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < normDim; i += blockDim.x)
    {
        float v = __half2float(inRow[i]);
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
        outRow[i] = __float2half(__half2float(inRow[i]) * invRms * weight[i]);
}

// ── Broadcast affine over the last dim (F16 activation, F32 scale/shift) ─────
// out[b,s,d] = in[b,s,d] * scale[b,d] + (shift ? shift[b,d] : 0). scale/shift broadcast over S.
__global__ void dit_affine_broadcast_lastdim_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
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

    float v = __half2float(input[i]) * scale[pIdx];
    if (shift != 0) v += shift[pIdx];
    output[i] = __float2half(v);
}

// ── Gated residual over the last dim (F16 activations, F32 gate) ────────────
// out[b,s,d] = residual[b,s,d] + gate[b,d] * value[b,s,d]. gate broadcast over S.
__global__ void dit_gated_residual_lastdim_f16(
    __half* __restrict__ output,
    const __half* __restrict__ residual,
    const __half* __restrict__ value,
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

    output[i] = __float2half(__half2float(residual[i]) + gate[pIdx] * __half2float(value[i]));
}

// ── Add scalar (out = in + c) ───────────────────────────────────────────────
__global__ void dit_add_scalar_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    float c,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = __float2half(__half2float(input[i]) + c);
}

// ── Sigmoid (F16 I/O, sign-aware F32 exp) ───────────────────────────────────
__global__ void dit_sigmoid_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = __half2float(input[i]);
    float y;
    if (x >= 0.0f) { float ex = expf(-x); y = 1.0f / (1.0f + ex); }
    else           { float ex = expf(x);  y = ex / (1.0f + ex); }
    output[i] = __float2half(y);
}

// ── Rotate-half RoPE (in-place, F16 x, F32 cos/sin) ─────────────────────────
// Mirrors dit_rope_f32: x is [totalVecs, headDim] (totalVecs = B*L*numHeads); cos/sin are
// [B*L, headDim] broadcast over heads. Pairs (i, i+half); each thread owns one i and writes
// both halves from the originals — race-free. rotaryDim semantics identical to the F32 twin.
__global__ void dit_rope_f16(
    __half* __restrict__ x,
    const float* __restrict__ cosT,
    const float* __restrict__ sinT,
    unsigned int numHeads,
    unsigned int headDim,
    unsigned long long totalVecs,
    unsigned int rotaryDim)
{
    unsigned int rdim = (rotaryDim == 0u || rotaryDim > headDim) ? headDim : rotaryDim;
    unsigned int half = rdim >> 1;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = totalVecs * (unsigned long long)half;
    if (gid >= total) return;

    unsigned int i = (unsigned int)(gid % half);
    unsigned long long vec = gid / half;
    unsigned long long row = vec / numHeads;
    size_t baseX = (size_t)vec * headDim;
    size_t baseCs = (size_t)row * headDim;

    float lower = __half2float(x[baseX + i]);
    float upper = __half2float(x[baseX + i + half]);
    x[baseX + i] = __float2half(lower * cosT[baseCs + i] - upper * sinT[baseCs + i]);
    x[baseX + i + half] = __float2half(upper * cosT[baseCs + i + half] + lower * sinT[baseCs + i + half]);
}

// ── Fused-lastdim slice (F16, pure copy) ────────────────────────────────────
// Mirrors dit_slice_lastdim_f32: out[row, d] = in[row, offset + d] with in row stride = inDim.
// Splits a fused [.., inDim] tensor (e.g. QKV [B,L,3H]) into a contiguous [.., outDim] chunk.
__global__ void dit_slice_lastdim_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    unsigned int outDim,
    unsigned int inDim,
    unsigned int offset,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int d = (unsigned int)(i % outDim);
    unsigned long long row = i / outDim;
    output[i] = input[(size_t)row * inDim + offset + d];
}

// ── Interleaved RoPE (in-place, F16 x, F32 cos/sin) ─────────────────────────
// Mirrors wan_rope_interleaved: x is [S, heads*headDim]; rotate adjacent pair (2i,2i+1) by the
// shared angle at cos/sin index 2i (cos/sin are [S, headDim] shared across heads).
__global__ void dit_rope_interleaved_f16(
    __half* __restrict__ x,
    const float* __restrict__ cosT,
    const float* __restrict__ sinT,
    unsigned int S, unsigned int heads, unsigned int headDim)
{
    unsigned int pairs = headDim >> 1;
    unsigned long long total = (unsigned long long)S * heads * pairs;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned int i = (unsigned int)(idx % pairs); idx /= pairs;
    unsigned int h = (unsigned int)(idx % heads); idx /= heads;
    unsigned int s = (unsigned int)idx;
    unsigned int i0 = 2u * i;
    unsigned long long xoff = ((unsigned long long)s * heads + h) * headDim + i0;
    unsigned long long coff = (unsigned long long)s * headDim + i0;   // shared across heads
    float re = __half2float(x[xoff]), im = __half2float(x[xoff + 1]);
    float c = cosT[coff], sn = sinT[coff];
    x[xoff]     = __float2half(re * c - im * sn);
    x[xoff + 1] = __float2half(re * sn + im * c);
}

// ── Slice a contiguous row block (F16, pure copy) ───────────────────────────
// output[i] = input[elemOffset + i]. Extracts the image-token rows from the joint [text,image] sequence.
__global__ void dit_slice_rows_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    unsigned long long elemOffset,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    output[i] = input[elemOffset + i];
}

// ── GQA K/V head repeat (F16, pure copy) ────────────────────────────────────
// out head qh = h*group + g maps to input head qh/group. One thread per output element. No math.
__global__ void dit_repeat_kv_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    unsigned int kvHeads,
    unsigned int group,
    unsigned int seqLen,
    unsigned int headDim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int d = (unsigned int)(i % headDim);
    unsigned long long rem = i / headDim;
    unsigned int l = (unsigned int)(rem % seqLen);
    rem /= seqLen;
    unsigned int qHeads = kvHeads * group;
    unsigned int qh = (unsigned int)(rem % qHeads);
    unsigned long long b = rem / qHeads;

    unsigned int inH = qh / group;
    unsigned long long inIdx = (((b * kvHeads + inH) * seqLen) + l) * (unsigned long long)headDim + d;
    output[i] = input[inIdx];
}

// ── Non-affine LayerNorm (F16 I/O, F32 accumulate) ──────────────────────────
// F16 twin of dit_layernorm_noaffine_f32: per row, normalize to zero mean / unit variance
// (biased var, /dim), no scale/bias. One block per row; shared-mem reduction.
__global__ void dit_layernorm_noaffine_f16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    unsigned int dim,
    unsigned int totalRows,
    float eps)
{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x;
    if (row >= totalRows) return;

    const __half* inRow = input + (size_t)row * dim;
    __half* outRow = output + (size_t)row * dim;

    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        partial += __half2float(inRow[i]);
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)dim;
    __syncthreads();

    float vpart = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float diff = __half2float(inRow[i]) - mean;
        vpart += diff * diff;
    }
    sdata[threadIdx.x] = vpart;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float invStd = rsqrtf(sdata[0] / (float)dim + eps);

    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
        outRow[i] = __float2half((__half2float(inRow[i]) - mean) * invStd);
}

// ── CHW F32 [-1,1] → HWC u8 [0,255] (image output conversion) ───────────────
// One thread per pixel; reads the 3 channel planes, writes the interleaved byte triple. Replaces the
// host loop in ImagePostProcessor.TensorToRgbBytes (12 MB D2H + 12M-element CPU loop → 3 MB D2H).
__global__ void dit_chw_f32_to_hwc_u8(
    unsigned char* __restrict__ output,
    const float* __restrict__ input,
    unsigned int height,
    unsigned int width)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;   // pixel index
    unsigned int hw = height * width;
    if (i >= hw) return;
    unsigned int outBase = i * 3u;
    for (int c = 0; c < 3; c++)
    {
        float v = (input[(unsigned long long)c * hw + i] + 1.0f) * 0.5f;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        output[outBase + c] = (unsigned char)(v * 255.0f + 0.5f);
    }
}

} // extern "C"
