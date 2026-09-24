// MXFP8 (ComfyUI `mxfp8`) resident weight -> F16/BF16: F8E4M3 [rows, cols] × one UE8M0 scale per 32 input elements in
// NVIDIA's blocked layout × the scale tensor's own Fp8ScaleFactor, so a checkpoint quantized this way stays resident at
// one byte per parameter and unpacks transiently per GEMM on cards without block-scaled tensor cores. Same decode as
// HartsyInference.Core.Tensors.Mxfp8ResidentCodec: an e4m3 value times a power of two is exact in F32, so the F16
// store rounds once and the BF16 store truncates to match Tensor.CastTo bit for bit.

#include <cuda_fp16.h>
#include "block_scale.cuh"

#define MXFP8_DEQUANT_BODY(STORE)                                                                          \
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)                                       \
    {                                                                                                       \
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;                                           \
        if (col >= cols) continue;                                                                          \
        float scale = e8m0_decode(blockScale[swizzled_scale_index(row, col >> 5, paddedCols)]) * scaleFactor; \
        unsigned long long i = (unsigned long long)row * cols + col;                                        \
        STORE(i, nvfp4_e4m3_decode(w[i]) * scale);                                                          \
    }

#define MXFP8_STORE_F16(I, V)  out[(I)] = __float2half(V)
#define MXFP8_STORE_BF16(I, V) out[(I)] = (unsigned short)(__float_as_uint(V) >> 16)

extern "C" __global__ void dequant_mxfp8_to_f16(
    const unsigned char* __restrict__ w, const unsigned char* __restrict__ blockScale, __half* __restrict__ out,
    unsigned int rows, unsigned int cols, unsigned int paddedCols, float scaleFactor)
{
    MXFP8_DEQUANT_BODY(MXFP8_STORE_F16)
}

extern "C" __global__ void dequant_mxfp8_to_bf16(
    const unsigned char* __restrict__ w, const unsigned char* __restrict__ blockScale, unsigned short* __restrict__ out,
    unsigned int rows, unsigned int cols, unsigned int paddedCols, float scaleFactor)
{
    MXFP8_DEQUANT_BODY(MXFP8_STORE_BF16)
}
