// The generic gated linear unit (IBackend.Glu) and group gate (IBackend.GroupScale) kernels.
//
//   glu_<t>                 out[r, i] = act(gate[r, i]) · value[r, i] over the [gate | value] or [value | gate] halves of each
//                           input row; act 0 SiLU, 1 GELU-tanh (the fast sigmoid form of lm_glu_act_f32), 2 GELU-erf,
//                           3 SiTU (4·tanh(g/4)·sigmoid(g) with the value soft-capped to 25·tanh(u/25)).
//   group_scale_<t>[_<g>]   out[i] = f(gate(row(i), group(i))) · in[i], or out[i] += that with a zero gate skipped; f is the
//                           stored gate (0) or 2·sigmoid (1). The output may be the input.
//
// The geometry is one by-value parameter (mirrored field for field by CudaKernels.Gating.cs), so a launch captures into a graph.
// Contracts and host references: GluContract / GluReference, GroupScaleContract / GroupScaleReference.
//
// Build: ./build.sh (nvrtc through ../build_common.sh).

#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct GluGeom
{
    unsigned long long rows;
    int inner;
    int act;
    int gateFirst;
};

struct GroupScaleGeom
{
    unsigned long long total;
    long long gateStride;
    long long gateOffset;
    int width;
    int groupWidth;
    int function;
    int accumulate;
};

__device__ __forceinline__ float load_act(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float load_act(const __half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float load_act(const __nv_bfloat16* p, size_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void store_act(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store_act(__half* p, size_t i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void store_act(__nv_bfloat16* p, size_t i, float v) { p[i] = __float2bfloat16(v); }

// The sign-aware logistic: neither tail evaluates exp of a large positive argument.
__device__ __forceinline__ float sigmoid_safe(float x)
{
    return x >= 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

__device__ __forceinline__ float glu_gate(float g, int act)
{
    if (act == 0) return g * __fdividef(1.0f, 1.0f + __expf(-g));                         // SiLU
    if (act == 1)
    {
        const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        const float s = __fdividef(1.0f, 1.0f + __expf(-2.0f * inner));                     // tanh(x) = 2·sigmoid(2x) − 1
        return 0.5f * g * (1.0f + (2.0f * s - 1.0f));                                       // GELU-tanh
    }
    if (act == 2) return 0.5f * g * (1.0f + erff(g * 0.70710678118654752440f));            // GELU-erf
    return 4.0f * tanhf(g * 0.25f) * sigmoid_safe(g);                                       // SiTU gate
}

__device__ __forceinline__ float glu_value(float u, int act)
{
    return act == 3 ? 25.0f * tanhf(u / 25.0f) : u;
}

template <typename T>
__device__ __forceinline__ void glu(T* __restrict__ out, const T* __restrict__ in, const GluGeom g)
{
    unsigned long long total = g.rows * (unsigned long long)g.inner;
    for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x)
    {
        unsigned long long r = idx / (unsigned long long)g.inner;
        unsigned int i = (unsigned int)(idx % (unsigned long long)g.inner);
        size_t base = (size_t)r * 2u * (size_t)g.inner;
        size_t gateAt = base + (g.gateFirst ? i : (size_t)g.inner + i);
        size_t valueAt = base + (g.gateFirst ? (size_t)g.inner + i : i);
        store_act(out, idx, glu_gate(load_act(in, gateAt), g.act) * glu_value(load_act(in, valueAt), g.act));
    }
}

template <typename T, typename TG>
__device__ __forceinline__ void group_scale(T* out, const T* in, const TG* __restrict__ gates, const GroupScaleGeom g)
{
    for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; idx < g.total; idx += (unsigned long long)gridDim.x * blockDim.x)
    {
        unsigned long long row = idx / (unsigned long long)g.width;
        int group = (int)(idx % (unsigned long long)g.width) / g.groupWidth;
        float gate = load_act(gates, (size_t)(g.gateOffset + (long long)row * g.gateStride + group));
        float factor = g.function == 1 ? 2.0f * sigmoid_safe(gate) : gate;
        if (g.accumulate)
        {
            if (factor != 0.0f) store_act(out, idx, load_act(out, idx) + factor * load_act(in, idx));
        }
        else
        {
            store_act(out, idx, factor * load_act(in, idx));
        }
    }
}

extern "C" {

__global__ void glu_f32(float* __restrict__ out, const float* __restrict__ in, const GluGeom g) { glu<float>(out, in, g); }
__global__ void glu_f16(__half* __restrict__ out, const __half* __restrict__ in, const GluGeom g) { glu<__half>(out, in, g); }
__global__ void glu_bf16(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in, const GluGeom g) { glu<__nv_bfloat16>(out, in, g); }

__global__ void group_scale_f32(float* out, const float* in, const float* __restrict__ gates, const GroupScaleGeom g) { group_scale<float, float>(out, in, gates, g); }
__global__ void group_scale_f16_f32(__half* out, const __half* in, const float* __restrict__ gates, const GroupScaleGeom g) { group_scale<__half, float>(out, in, gates, g); }
__global__ void group_scale_f16_f16(__half* out, const __half* in, const __half* __restrict__ gates, const GroupScaleGeom g) { group_scale<__half, __half>(out, in, gates, g); }
__global__ void group_scale_bf16_f32(__nv_bfloat16* out, const __nv_bfloat16* in, const float* __restrict__ gates, const GroupScaleGeom g) { group_scale<__nv_bfloat16, float>(out, in, gates, g); }
__global__ void group_scale_bf16_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, const __nv_bfloat16* __restrict__ gates, const GroupScaleGeom g) { group_scale<__nv_bfloat16, __nv_bfloat16>(out, in, gates, g); }

}
