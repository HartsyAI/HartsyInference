// Channels-last depthwise k x k convolution fused with a GLU (IBackend.DepthwiseConvGlu), F32 and BF16 activations.
//
//   input  [frames, height, width, 2H]  value channels [0, H), gate channels [H, 2H); optionally SiLU'd as it is read
//   taps   [k, k, 2H] F32 (tap-major: the taps of consecutive channels are contiguous)
//   output [frames, height, width, H]   (value conv + bias) * SiLU(gate conv + bias)
//
// The 2H-wide convolution output is never stored, and with inputSilu the activation of the producing GEMM is applied on the
// load instead of in a separate pass. Host reference: DepthwiseConvGluReference.
//
//   depthwise_glu_{f32,bf16}        one thread per (pixel, output channel); any shape
//   depthwise_glu_bf16_vec          one thread per 8 channels and a row of pixels, walked in runs of 4: 16-byte loads, and
//                                   each loaded column serves up to three outputs of the run (k = 3, H % 8 == 0, W % 4 == 0)
//
// Build: src/HartsyInference.Cuda/Kernels/conv/build.sh.

#include <cuda_bf16.h>

#define DG_VEC 8
#define DG_RUN 4

__device__ __forceinline__ float dg_silu(float v) { return __fdividef(v, 1.0f + __expf(-v)); }

template <typename T> __device__ __forceinline__ float dg_load(const T* p);
template <> __device__ __forceinline__ float dg_load<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float dg_load<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ __forceinline__ T dg_store(float v);
template <> __device__ __forceinline__ float dg_store<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 dg_store<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ void depthwise_glu(T* __restrict__ out, const T* __restrict__ in, const float* __restrict__ taps,
    const float* __restrict__ bias, unsigned int frames, unsigned int height, unsigned int width, unsigned int hidden,
    unsigned int kernel, int inputSilu)
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
            const float* tap = taps + (unsigned long long)(ky * kernel + kx) * channels;
            float v = dg_load<T>(src + c), g = dg_load<T>(src + hidden + c);
            if (inputSilu) { v = dg_silu(v); g = dg_silu(g); }
            value += tap[c] * v;
            gate += tap[hidden + c] * g;
        }
    }
    out[pixel * hidden + c] = dg_store<T>(value * dg_silu(gate));
}

__device__ __forceinline__ void dg_unpack(uint4 raw, float* v, int silu)
{
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = silu ? dg_silu(f.x) : f.x;
        v[2 * i + 1] = silu ? dg_silu(f.y) : f.y;
    }
}

__device__ __forceinline__ void dg_taps(const float* p, float* t)
{
    float4 a = *reinterpret_cast<const float4*>(p);
    float4 b = *reinterpret_cast<const float4*>(p + 4);
    t[0] = a.x; t[1] = a.y; t[2] = a.z; t[3] = a.w; t[4] = b.x; t[5] = b.y; t[6] = b.z; t[7] = b.w;
}

extern "C" {

__global__ void depthwise_glu_f32(float* __restrict__ out, const float* __restrict__ in, const float* __restrict__ taps,
    const float* __restrict__ bias, unsigned int frames, unsigned int height, unsigned int width, unsigned int hidden,
    unsigned int kernel, int inputSilu)
{
    depthwise_glu<float>(out, in, taps, bias, frames, height, width, hidden, kernel, inputSilu);
}

__global__ void depthwise_glu_bf16(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ taps, const float* __restrict__ bias, unsigned int frames, unsigned int height,
    unsigned int width, unsigned int hidden, unsigned int kernel, int inputSilu)
{
    depthwise_glu<__nv_bfloat16>(out, in, taps, bias, frames, height, width, hidden, kernel, inputSilu);
}

// Accumulates one half (value or gate) of a DG_RUN-pixel run: each input column of the three input rows is loaded once and
// added into every output of the run that reads it (up to three).
__device__ __forceinline__ void dg_run_half(float (*acc)[DG_VEC], const __nv_bfloat16* __restrict__ in, const float* __restrict__ taps,
    const float* bias8, unsigned long long f, unsigned int y, unsigned int x0, unsigned int height, unsigned int width,
    unsigned int channels, unsigned int c, int inputSilu)
{
#pragma unroll
    for (int i = 0; i < DG_RUN; i++)
#pragma unroll
        for (int k = 0; k < DG_VEC; k++) acc[i][k] = bias8[k];

#pragma unroll
    for (int ky = 0; ky < 3; ky++)
    {
        const int sy = (int)y + ky - 1;
        if (sy < 0 || sy >= (int)height) continue;
        float t[3][DG_VEC];
#pragma unroll
        for (int kx = 0; kx < 3; kx++) dg_taps(taps + (unsigned long long)(ky * 3 + kx) * channels + c, t[kx]);
        const unsigned long long rowBase = (f * height + (unsigned long long)sy) * width;
#pragma unroll
        for (int col = 0; col < DG_RUN + 2; col++)
        {
            const int sx = (int)x0 + col - 1;
            if (sx < 0 || sx >= (int)width) continue;
            float v[DG_VEC];
            dg_unpack(*reinterpret_cast<const uint4*>(in + (rowBase + (unsigned long long)sx) * channels + c), v, inputSilu);
#pragma unroll
            for (int i = 0; i < DG_RUN; i++)
            {
                const int kx = col - i;
                if (kx < 0 || kx > 2) continue;
#pragma unroll
                for (int k = 0; k < DG_VEC; k++) acc[i][k] += t[kx][k] * v[k];
            }
        }
    }
}

// grid (ceil(H/8 / blockDim.x), frames*height); thread = 8 channels of one image row, walked in runs of DG_RUN pixels.
__global__ void depthwise_glu_bf16_vec(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ taps, const float* __restrict__ bias, unsigned int frames, unsigned int height,
    unsigned int width, unsigned int hidden, int inputSilu)
{
    const unsigned int vecs = hidden / DG_VEC;
    const unsigned int cv = blockIdx.x * blockDim.x + threadIdx.x;
    if (cv >= vecs) return;
    const unsigned int row = blockIdx.y;
    const unsigned int y = row % height;
    const unsigned long long f = row / height;
    const unsigned int c0 = cv * DG_VEC;
    const unsigned int channels = 2u * hidden;

    float biasV[DG_VEC], biasG[DG_VEC];
    if (bias) { dg_taps(bias + c0, biasV); dg_taps(bias + hidden + c0, biasG); }
    else {
#pragma unroll
        for (int i = 0; i < DG_VEC; i++) { biasV[i] = 0.0f; biasG[i] = 0.0f; }
    }

    for (unsigned int x0 = 0; x0 < width; x0 += DG_RUN)
    {
        float va[DG_RUN][DG_VEC], ga[DG_RUN][DG_VEC];
        dg_run_half(va, in, taps, biasV, f, y, x0, height, width, channels, c0, inputSilu);
        dg_run_half(ga, in, taps, biasG, f, y, x0, height, width, channels, hidden + c0, inputSilu);
#pragma unroll
        for (int i = 0; i < DG_RUN; i++)
        {
            uint4 packed;
            __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
            for (int k = 0; k < DG_VEC / 2; k++)
                h[k] = __floats2bfloat162_rn(va[i][2 * k] * dg_silu(ga[i][2 * k]), va[i][2 * k + 1] * dg_silu(ga[i][2 * k + 1]));
            const unsigned long long pixel = (f * height + y) * width + x0 + i;
            *reinterpret_cast<uint4*>(out + pixel * hidden + c0) = packed;
        }
    }
}

} // extern "C"
