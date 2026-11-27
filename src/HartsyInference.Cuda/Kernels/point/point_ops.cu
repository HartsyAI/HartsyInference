// The per-element and per-point ops: uniform fake quantization (IBackend.FakeQuantizeUniform), the Fourier feature embedding
// (IBackend.FourierEmbed) and elementwise exp and reciprocal (IBackend.Exp, IBackend.Reciprocal). By-value parameters only, so
// a launch captures into a graph. F32, except the elementwise math, which reads and writes F32 or BF16 and computes in F32.
//
//   fake_quantize_uniform_f32  out[i] = low + trunc((clamp(in[i], low, high) − low) / range · steps) / steps · range
//   fourier_embed_f32          per point and coordinate: the coordinate (when included), then sin(x · 2^b) and cos(x · 2^b)
//                              coordinate-major over the bands; one thread per (point, coordinate)
//   elementwise_math_<in>_<out> out[i] = expf(in[i]) (function 0) or 1.0f / in[i] (function 1); BF16 rounds to nearest even
//
// Contracts and host references: FakeQuantizeContract / FakeQuantizeReference, FourierEmbedContract / FourierEmbedReference,
// ElementwiseMathContract / ElementwiseMathReference.
// Build: ./build.sh (nvrtc through ../build_common.sh).

#include <cuda_bf16.h>

struct FakeQuantizeArgs
{
    float low;
    float high;
    float range;
    float steps;
    unsigned long long count;
};

struct FourierEmbedArgs
{
    unsigned long long count;
    unsigned int dims;
    unsigned int bands;
    unsigned int width;
    unsigned int includeInput;
};

extern "C" __global__ void fake_quantize_uniform_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    FakeQuantizeArgs a)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.count) return;
    float v = input[i];
    v = v > a.high ? a.high : (v < a.low ? a.low : v);
    int b = (int)((v - a.low) / a.range * a.steps);
    output[i] = (float)b / a.steps * a.range + a.low;
}

extern "C" __global__ void fourier_embed_f32(
    float* __restrict__ dst,
    const float* __restrict__ coords,
    FourierEmbedArgs a)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= a.count * a.dims) return;
    unsigned int c = (unsigned int)(gid % a.dims);
    unsigned long long i = gid / a.dims;
    float x = coords[i * a.dims + c];
    unsigned long long o = i * a.width;
    unsigned int sinBase = a.includeInput ? a.dims : 0u, cosBase = sinBase + a.dims * a.bands;
    if (a.includeInput) dst[o + c] = x;
    for (unsigned int band = 0; band < a.bands; band++)
    {
        float v = x * (float)(1u << band);
        dst[o + sinBase + c * a.bands + band] = sinf(v);
        dst[o + cosBase + c * a.bands + band] = cosf(v);
    }
}

struct ElementwiseMathArgs
{
    unsigned long long count;
    unsigned int function;
    unsigned int reserved;
};

__device__ __forceinline__ float load_math_operand(const float* p, unsigned long long i) { return p[i]; }
__device__ __forceinline__ float load_math_operand(const __nv_bfloat16* p, unsigned long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void store_math_result(float* p, unsigned long long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_math_result(__nv_bfloat16* p, unsigned long long i, float v) { p[i] = __float2bfloat16(v); }

template <typename TIn, typename TOut>
__device__ __forceinline__ void elementwise_math(TOut* __restrict__ output, const TIn* __restrict__ input, ElementwiseMathArgs a)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.count) return;
    float x = load_math_operand(input, i);
    store_math_result(output, i, a.function == 0u ? expf(x) : 1.0f / x);
}

extern "C" __global__ void elementwise_math_f32_f32(float* __restrict__ output, const float* __restrict__ input, ElementwiseMathArgs a)
{
    elementwise_math(output, input, a);
}

extern "C" __global__ void elementwise_math_f32_bf16(__nv_bfloat16* __restrict__ output, const float* __restrict__ input, ElementwiseMathArgs a)
{
    elementwise_math(output, input, a);
}

extern "C" __global__ void elementwise_math_bf16_f32(float* __restrict__ output, const __nv_bfloat16* __restrict__ input, ElementwiseMathArgs a)
{
    elementwise_math(output, input, a);
}

extern "C" __global__ void elementwise_math_bf16_bf16(__nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ input,
    ElementwiseMathArgs a)
{
    elementwise_math(output, input, a);
}
