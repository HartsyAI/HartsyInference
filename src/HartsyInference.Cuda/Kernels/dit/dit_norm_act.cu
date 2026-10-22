// Fused DiT norm/activation kernels — F32 compute, F32 or F16 activations.
//
//   rms_norm_modulate_<in>_<out>  out[r,d] = x[r,d]·rsqrt(mean(x[r,:]²) + eps)·w[d]·(1 + scale[m,d]) + shift[m,d]
//                                 One block per row; m = r / rowsPerMod picks the modulation row. In and out
//                                 dtypes are independent so an F32 residual stream can feed an F16 branch.
//   layer_norm_modulate_f32_f16   out[r,d] = (x[r,d] − mean)·rsqrt(var + eps)·w[d]·(1 + scale[m,d]) + shift[m,d]
//                                 over an F32 row into F16; a null w, scale or shift drops its term.
//   gated_residual_f32_f16        out[i] = residual[i] + gate[m,d]·value[i]: an F16 branch added to an F32 stream.
//   situ_glu_<t>                  out[r,j] = (4·tanh(g/4)·sigmoid(g)) · (25·tanh(u/25)) over in[r] = [u | g].
//
// Contracts and host references: IBackend.RmsNormModulate / RmsNormModulateReference, IBackend.LayerNorm and
// LayerNormModulate, IBackend.GatedResidualLastDim, and IBackend.SituGlu / SituGluReference.
//
// Build: src/HartsyInference.Cuda/Kernels/dit/build.sh (nvcc, or the nvrtc helper when no toolkit is present).

#include <cuda_fp16.h>

__device__ __forceinline__ float load_act(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float load_act(const __half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void store_act(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store_act(__half* p, size_t i, float v) { p[i] = __float2half(v); }

// Block-wide sum; blockDim.x is a multiple of 32 and at most 1024.
__device__ __forceinline__ float block_sum(float v)
{
    __shared__ float warpSums[32];
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();
    unsigned int warps = (blockDim.x + 31u) >> 5;
    v = lane < warps ? warpSums[lane] : 0.0f;
    if (warp == 0)
    {
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
        if (lane == 0) warpSums[0] = v;
    }
    __syncthreads();
    float total = warpSums[0];
    __syncthreads();
    return total;
}

template <typename TIn, typename TOut>
__device__ __forceinline__ void rms_norm_modulate(
    TOut* __restrict__ out, const TIn* __restrict__ in, const float* __restrict__ weight,
    const float* __restrict__ scale, const float* __restrict__ shift,
    unsigned int dim, unsigned int rows, unsigned int rowsPerMod, float eps)
{
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    size_t base = (size_t)r * dim;
    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float v = load_act(in, base + i);
        partial += v * v;
    }
    float inv = rsqrtf(block_sum(partial) / (float)dim + eps);
    size_t mod = (size_t)(r / rowsPerMod) * dim;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float v = load_act(in, base + i) * inv * weight[i] * (1.0f + scale[mod + i]);
        if (shift != 0) v += shift[mod + i];
        store_act(out, base + i, v);
    }
}

template <typename TIn, typename TOut>
__device__ __forceinline__ void layer_norm_modulate(
    TOut* __restrict__ out, const TIn* __restrict__ in, const float* __restrict__ weight,
    const float* __restrict__ scale, const float* __restrict__ shift,
    unsigned int dim, unsigned int rows, unsigned int rowsPerMod, float eps)
{
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    size_t base = (size_t)r * dim;
    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) partial += load_act(in, base + i);
    float mean = block_sum(partial) / (float)dim;
    float vpart = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float d = load_act(in, base + i) - mean;
        vpart += d * d;
    }
    float inv = rsqrtf(block_sum(vpart) / (float)dim + eps);
    size_t mod = (size_t)(r / rowsPerMod) * dim;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float v = (load_act(in, base + i) - mean) * inv;
        if (weight != 0) v *= weight[i];
        if (scale != 0) v *= 1.0f + scale[mod + i];
        if (shift != 0) v += shift[mod + i];
        store_act(out, base + i, v);
    }
}

__device__ __forceinline__ float situ(float value, float gate)
{
    float sig = gate >= 0.0f ? 1.0f / (1.0f + expf(-gate)) : expf(gate) / (1.0f + expf(gate));
    return 4.0f * tanhf(gate * 0.25f) * sig * (25.0f * tanhf(value / 25.0f));
}

template <typename T>
__device__ __forceinline__ void situ_glu(T* __restrict__ out, const T* __restrict__ in, unsigned int inner,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned long long row = i / inner;
    unsigned int j = (unsigned int)(i - row * inner);
    size_t at = (size_t)row * 2u * inner + j;
    store_act(out, i, situ(load_act(in, at), load_act(in, at + inner)));
}

extern "C" {

__global__ void rms_norm_modulate_f32_f32(float* out, const float* in, const float* weight, const float* scale,
    const float* shift, unsigned int dim, unsigned int rows, unsigned int rowsPerMod, float eps)
{
    rms_norm_modulate(out, in, weight, scale, shift, dim, rows, rowsPerMod, eps);
}

__global__ void rms_norm_modulate_f32_f16(__half* out, const float* in, const float* weight, const float* scale,
    const float* shift, unsigned int dim, unsigned int rows, unsigned int rowsPerMod, float eps)
{
    rms_norm_modulate(out, in, weight, scale, shift, dim, rows, rowsPerMod, eps);
}

__global__ void rms_norm_modulate_f16_f16(__half* out, const __half* in, const float* weight, const float* scale,
    const float* shift, unsigned int dim, unsigned int rows, unsigned int rowsPerMod, float eps)
{
    rms_norm_modulate(out, in, weight, scale, shift, dim, rows, rowsPerMod, eps);
}

__global__ void rms_norm_modulate_f16_f32(float* out, const __half* in, const float* weight, const float* scale,
    const float* shift, unsigned int dim, unsigned int rows, unsigned int rowsPerMod, float eps)
{
    rms_norm_modulate(out, in, weight, scale, shift, dim, rows, rowsPerMod, eps);
}

__global__ void situ_glu_f32(float* out, const float* in, unsigned int inner, unsigned long long total)
{
    situ_glu(out, in, inner, total);
}

__global__ void situ_glu_f16(__half* out, const __half* in, unsigned int inner, unsigned long long total)
{
    situ_glu(out, in, inner, total);
}

__global__ void layer_norm_modulate_f32_f16(__half* out, const float* in, const float* weight, const float* scale,
    const float* shift, unsigned int dim, unsigned int rows, unsigned int rowsPerMod, float eps)
{
    layer_norm_modulate(out, in, weight, scale, shift, dim, rows, rowsPerMod, eps);
}

// The gate row is b = row / seqLen, as in dit_gated_residual_lastdim_f32; the F16 value widens exactly.
__global__ void gated_residual_f32_f16(float* __restrict__ output, const float* __restrict__ residual,
    const __half* __restrict__ value, const float* __restrict__ gate, unsigned int seqLen, unsigned int dim,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int d = (unsigned int)(i % dim);
    unsigned long long row = i / dim;
    unsigned int b = (unsigned int)(row / seqLen);
    output[i] = residual[i] + gate[(size_t)b * dim + d] * __half2float(value[i]);
}

}
