// Channels-last depthwise k x k convolution fused with a GLU (IBackend.DepthwiseConvGlu), F32 and BF16 activations.
//
//   input  [frames, height, width, 2H]  value channels [0, H), gate channels [H, 2H)
//   output [frames, height, width, H]   (value conv + bias) * SiLU(gate conv + bias)
//
// One thread per (pixel, output channel): it sums the k*k taps of its value and gate channels in F32 and writes the
// product once, so the 2H-wide convolution output is never stored. Consecutive threads take consecutive channels of a
// pixel, so every tap load is coalesced. Taps are F32 [2H, k*k]; the bias is F32 [2H] or absent. Host reference:
// DepthwiseConvGluReference.
//
// Build: src/HartsyInference.Cuda/Kernels/conv/build.sh.

#include <cuda_bf16.h>

template <typename T> __device__ __forceinline__ float dg_load(const T* p);
template <> __device__ __forceinline__ float dg_load<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float dg_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ __forceinline__ T dg_store(float v);
template <> __device__ __forceinline__ float dg_store<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 dg_store<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void depthwise_glu(T* __restrict__ out, const T* __restrict__ in, const float* __restrict__ taps,
    const float* __restrict__ bias, unsigned int frames, unsigned int height, unsigned int width, unsigned int hidden,
    unsigned int kernel)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)frames * height * width * hidden;
    if (idx >= total) return;
    const unsigned int c = (unsigned int)(idx % hidden);
    const unsigned long long pixel = idx / hidden;
    const unsigned int x = (unsigned int)(pixel % width);
    const unsigned long long fy = pixel / width;
    const unsigned int y = (unsigned int)(fy % height);
    const unsigned long long f = fy / height;
    const unsigned int channels = 2u * hidden;
    const int pad = (int)kernel / 2;
    const unsigned int area = kernel * kernel;
    const float* tv = taps + (unsigned long long)c * area;
    const float* tg = taps + (unsigned long long)(hidden + c) * area;

    float value = bias ? bias[c] : 0.0f;
    float gate = bias ? bias[hidden + c] : 0.0f;
    for (unsigned int ky = 0; ky < kernel; ky++)
    {
        int sy = (int)y + (int)ky - pad;
        if (sy < 0 || sy >= (int)height) continue;
        for (unsigned int kx = 0; kx < kernel; kx++)
        {
            int sx = (int)x + (int)kx - pad;
            if (sx < 0 || sx >= (int)width) continue;
            const T* src = in + (((f * height + (unsigned long long)sy) * width + (unsigned long long)sx) * channels);
            unsigned int tap = ky * kernel + kx;
            value += tv[tap] * dg_load<T>(src + c);
            gate += tg[tap] * dg_load<T>(src + hidden + c);
        }
    }
    out[pixel * hidden + c] = dg_store<T>(value * (gate / (1.0f + __expf(-gate))));
}

extern "C" {

__global__ void depthwise_glu_f32(float* __restrict__ out, const float* __restrict__ in, const float* __restrict__ taps,
    const float* __restrict__ bias, unsigned int frames, unsigned int height, unsigned int width, unsigned int hidden,
    unsigned int kernel)
{
    depthwise_glu<float>(out, in, taps, bias, frames, height, width, hidden, kernel);
}

__global__ void depthwise_glu_bf16(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ taps, const float* __restrict__ bias, unsigned int frames, unsigned int height,
    unsigned int width, unsigned int hidden, unsigned int kernel)
{
    depthwise_glu<__nv_bfloat16>(out, in, taps, bias, frames, height, width, hidden, kernel);
}

} // extern "C"
