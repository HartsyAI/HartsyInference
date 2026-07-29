// Depthwise Conv2D: groups == channels, so each output channel convolves exactly one input channel.
// Weight layout is the fixed [C, 1, kH, kW] the IBackend.Conv2dDepthwise contract mandates; bias is
// [C] and optional (hasBias flag). One thread per output element, FP32 accumulation even for FP16 I/O.
// Matches the CPU reference in IBackend.Conv2dDepthwise exactly.
//
//   out[n,c,oy,ox] = bias[c] + sum_{ky,kw} w[c,0,ky,kw] * in[n,c, oy*strideH+ky-padH, ox*strideW+kw-padW]
//
// 64-bit thread indexing: N*C*oH*oW overflows u32 at high channel counts × 1024^2.
//
// Build (no nvcc on this box — use the repo's nvrtc frontend):
//   LD_LIBRARY_PATH=~/.local/lib/cuda13 src/HartsyInference.Cuda/Kernels/nvrtc_compile \
//     src/HartsyInference.Cuda/Kernels/conv/depthwise_conv2d.cu src/HartsyInference.Cuda/Kernels/conv/depthwise_conv2d.ptx compute_80 ~/.local/lib/cuda13/include
//   cp src/HartsyInference.Cuda/Kernels/conv/depthwise_conv2d.ptx src/HartsyInference.Cuda/Ptx/

#include <cuda_fp16.h>

extern "C" {

#define DEPTHWISE_CONV2D_BODY(T)                                                                   \
    unsigned long long total = (unsigned long long)n * channels * outH * outW;                     \
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;            \
    if (idx >= total) return;                                                                       \
    unsigned int ox = (unsigned int)(idx % outW);                                                   \
    unsigned long long t = idx / outW;                                                              \
    unsigned int oy = (unsigned int)(t % outH);                                                     \
    t /= outH;                                                                                       \
    unsigned int c = (unsigned int)(t % channels);                                                  \
    unsigned int b = (unsigned int)(t / channels);                                                  \
    const T* plane = input + ((size_t)(b * channels + c)) * inH * inW;                              \
    const T* kern = weight + (size_t)c * kH * kW;                                                    \
    int iy0 = (int)(oy * strideH) - (int)padH;                                                       \
    int ix0 = (int)(ox * strideW) - (int)padW;                                                       \
    float acc = hasBias ? (float)bias[c] : 0.0f;                                                     \
    for (unsigned int ky = 0; ky < kH; ky++) {                                                       \
        int iy = iy0 + (int)ky;                                                                      \
        if (iy < 0 || iy >= (int)inH) continue;                                                      \
        for (unsigned int kx = 0; kx < kW; kx++) {                                                   \
            int ix = ix0 + (int)kx;                                                                  \
            if (ix < 0 || ix >= (int)inW) continue;                                                  \
            acc += (float)kern[ky * kW + kx] * (float)plane[(size_t)iy * inW + (unsigned int)ix];    \
        }                                                                                            \
    }                                                                                                \
    output[idx] = (T)acc;

__global__ void depthwise_conv2d_f32(
    float* __restrict__ output, const float* __restrict__ input,
    const float* __restrict__ weight, const float* __restrict__ bias, int hasBias,
    unsigned int n, unsigned int channels, unsigned int inH, unsigned int inW,
    unsigned int outH, unsigned int outW,
    unsigned int kH, unsigned int kW,
    unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW)
{
    DEPTHWISE_CONV2D_BODY(float)
}

__global__ void depthwise_conv2d_f16(
    __half* __restrict__ output, const __half* __restrict__ input,
    const __half* __restrict__ weight, const __half* __restrict__ bias, int hasBias,
    unsigned int n, unsigned int channels, unsigned int inH, unsigned int inW,
    unsigned int outH, unsigned int outW,
    unsigned int kH, unsigned int kW,
    unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW)
{
    DEPTHWISE_CONV2D_BODY(__half)
}

} // extern "C"
