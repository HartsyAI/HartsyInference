// Banded im2col: same math and col layout as spatial_*.ptx im2col, restricted to an output-row band
// [ohBase, ohBase + bandRows). Caps Conv2D's im2col workspace at the band size so a single huge conv
// (e.g. a 512-ch 3x3 at 1024^2 output = 9.2 GB) runs as a few ~2 GB bands instead of one allocation
// that cannot fit next to resident model weights. Bit-identical: each band feeds the same GEMM with the
// output pointer offset to the band's rows (ldc stays the full outH*outW, so channel planes interleave
// exactly as in the unbanded call).
//
//   col[row = c*kH*kW + kh*kW + kw, (oh - ohBase)*outW + ow] = in[batchChanOffset + c, ih, iw]
//   ih = oh*strideH + kh - padH, iw = ow*strideW + kw - padW, zero outside.
//
// Build (no nvcc on this box — use the repo's nvrtc frontend):
//   LD_LIBRARY_PATH=~/.local/lib/cuda13 src/HartsyInference.Cuda/Kernels/nvrtc_compile \
//     src/HartsyInference.Cuda/Kernels/conv/im2col_banded.cu src/HartsyInference.Cuda/Kernels/conv/im2col_banded.ptx compute_80 ~/.local/lib/cuda13/include
//   cp src/HartsyInference.Cuda/Kernels/conv/im2col_banded.ptx src/HartsyInference.Cuda/Ptx/

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

#define IM2COL_BANDED_BODY(T)                                                                  \
    unsigned long long bandCols = (unsigned long long)bandRows * outW;                        \
    unsigned long long rows = (unsigned long long)channels * kH * kW;                         \
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;       \
    if (idx >= rows * bandCols) return;                                                        \
    unsigned long long row = idx / bandCols;                                                   \
    unsigned int colIdx = (unsigned int)(idx - row * bandCols);                                \
    unsigned int c = (unsigned int)(row / (kH * kW));                                          \
    unsigned int kIdx = (unsigned int)(row - (unsigned long long)c * kH * kW);                 \
    unsigned int kh = kIdx / kW;                                                               \
    unsigned int kw = kIdx - kh * kW;                                                          \
    unsigned int ohLocal = colIdx / outW;                                                      \
    unsigned int ow = colIdx - ohLocal * outW;                                                 \
    unsigned int oh = ohLocal + ohBase;                                                        \
    int ih = (int)(oh * strideH + kh) - (int)padH;                                             \
    int iw = (int)(ow * strideW + kw) - (int)padW;                                             \
    T val = (T)0.0f;                                                                           \
    if (ih >= 0 && ih < (int)inH && iw >= 0 && iw < (int)inW)                                  \
        val = input[(((size_t)batchChanOffset + c) * inH + (unsigned int)ih) * inW + (unsigned int)iw]; \
    col[row * bandCols + colIdx] = val;

__global__ void im2col_banded_f32(
    float* __restrict__ col, const float* __restrict__ input,
    unsigned int channels, unsigned int inH, unsigned int inW,
    unsigned int kH, unsigned int kW,
    unsigned int padH, unsigned int padW,
    unsigned int strideH, unsigned int strideW,
    unsigned int outW, unsigned int ohBase, unsigned int bandRows,
    unsigned int batchChanOffset)
{
    IM2COL_BANDED_BODY(float)
}

__global__ void im2col_banded_f16(
    __half* __restrict__ col, const __half* __restrict__ input,
    unsigned int channels, unsigned int inH, unsigned int inW,
    unsigned int kH, unsigned int kW,
    unsigned int padH, unsigned int padW,
    unsigned int strideH, unsigned int strideW,
    unsigned int outW, unsigned int ohBase, unsigned int bandRows,
    unsigned int batchChanOffset)
{
    IM2COL_BANDED_BODY(__half)
}

__global__ void im2col_banded_bf16(
    __nv_bfloat16* __restrict__ col, const __nv_bfloat16* __restrict__ input,
    unsigned int channels, unsigned int inH, unsigned int inW,
    unsigned int kH, unsigned int kW,
    unsigned int padH, unsigned int padW,
    unsigned int strideH, unsigned int strideW,
    unsigned int outW, unsigned int ohBase, unsigned int bandRows,
    unsigned int batchChanOffset)
{
    IM2COL_BANDED_BODY(__nv_bfloat16)
}

} // extern "C"
