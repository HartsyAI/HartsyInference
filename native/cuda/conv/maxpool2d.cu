// MaxPool2D: explicit kernel/stride/zero-pad 2D max-pooling over NCHW tensors. One thread per output
// element; the receptive field is scanned with bounds checks so padded taps never contribute. Matches
// the CPU reference in IBackend.MaxPool2D exactly, including the "entire window out-of-bounds → 0"
// fallback (unreachable for YOLO's SPPF k=5,s=1,p=2 but defined for arbitrary configs).
//
//   out[n,c,oy,ox] = max over (ky,kw) of in[n,c, oy*strideH+ky-padH, ox*strideW+kw-padW]
//
// 64-bit thread indexing: N*C*oH*oW overflows u32 at high channel counts × 1024^2.
//
// Build (no nvcc on this box — use the repo's nvrtc frontend):
//   LD_LIBRARY_PATH=~/.local/lib/cuda13 native/cuda/nvrtc_compile \
//     native/cuda/conv/maxpool2d.cu native/cuda/conv/maxpool2d.ptx compute_80 ~/.local/lib/cuda13/include
//   cp native/cuda/conv/maxpool2d.ptx src/HartsyInference.Cuda/Ptx/

#include <cuda_fp16.h>

extern "C" {

#define MAXPOOL2D_BODY(T)                                                                          \
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
    int iy0 = (int)(oy * strideH) - (int)padH;                                                       \
    int ix0 = (int)(ox * strideW) - (int)padW;                                                       \
    float maxVal = __int_as_float(0xff800000);  /* -inf, no math_constants.h dependency */          \
    for (unsigned int ky = 0; ky < kernelH; ky++) {                                                 \
        int iy = iy0 + (int)ky;                                                                      \
        if (iy < 0 || iy >= (int)inH) continue;                                                      \
        for (unsigned int kx = 0; kx < kernelW; kx++) {                                              \
            int ix = ix0 + (int)kx;                                                                  \
            if (ix < 0 || ix >= (int)inW) continue;                                                  \
            float v = (float)plane[(size_t)iy * inW + (unsigned int)ix];                             \
            if (v > maxVal) maxVal = v;                                                              \
        }                                                                                            \
    }                                                                                                \
    output[idx] = (T)(isinf(maxVal) && maxVal < 0.0f ? 0.0f : maxVal);

__global__ void maxpool2d_f32(
    float* __restrict__ output, const float* __restrict__ input,
    unsigned int n, unsigned int channels, unsigned int inH, unsigned int inW,
    unsigned int outH, unsigned int outW,
    unsigned int kernelH, unsigned int kernelW,
    unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW)
{
    MAXPOOL2D_BODY(float)
}

__global__ void maxpool2d_f16(
    __half* __restrict__ output, const __half* __restrict__ input,
    unsigned int n, unsigned int channels, unsigned int inH, unsigned int inW,
    unsigned int outH, unsigned int outW,
    unsigned int kernelH, unsigned int kernelW,
    unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW)
{
    MAXPOOL2D_BODY(__half)
}

} // extern "C"
