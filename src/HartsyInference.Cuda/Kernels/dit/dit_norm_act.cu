// Fused DiT activation kernels — F32 compute, F32 or F16 activations.
//
//   gated_residual_f32_f16        out[i] = residual[i] + gate[m,d]·value[i]: an F16 branch added to an F32 stream.
//   situ_glu_<t>                  out[r,j] = (4·tanh(g/4)·sigmoid(g)) · (25·tanh(u/25)) over in[r] = [u | g].
//
// Contracts and host references: IBackend.GatedResidualLastDim, and IBackend.SituGlu / SituGluReference.
//
// Build: src/HartsyInference.Cuda/Kernels/dit/build.sh (nvcc, or the nvrtc helper when no toolkit is present).

#include <cuda_fp16.h>

__device__ __forceinline__ float load_act(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float load_act(const __half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void store_act(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store_act(__half* p, size_t i, float v) { p[i] = __float2half(v); }

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

__global__ void situ_glu_f32(float* out, const float* in, unsigned int inner, unsigned long long total)
{
    situ_glu(out, in, inner, total);
}

__global__ void situ_glu_f16(__half* out, const __half* in, unsigned int inner, unsigned long long total)
{
    situ_glu(out, in, inner, total);
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
