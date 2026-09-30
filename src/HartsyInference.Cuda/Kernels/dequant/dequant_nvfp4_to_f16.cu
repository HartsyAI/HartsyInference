// NVFP4 (ComfyUI `nvfp4`) packed weight -> F16/BF16, so a checkpoint quantized this way stays RESIDENT at
// 0.5 byte/param and is dequantized transiently per GEMM instead of being unpacked to 16-bit at load. That is a
// VRAM win and nothing else: neither SM 8.6 nor SM 8.9 has FP4 tensor cores, so the GEMM still runs in F16/BF16
// off the dequantized copy this kernel writes. The saving is what makes the official 18.72 GB LTX-2.5 distilled
// DiT (42 GB once unpacked) fit a 24 GB card at all.
//
// Layout, matching HartsyInference.Core.Tensors.Nvfp4ResidentCodec:
//   weight        U8 [n, k/2], two E2M1 nibbles per byte, HIGH nibble = even element (opposite of MXFP4)
//   weight_scale  E4M3 [paddedRows, paddedCols], one scale per 16 input elements, in NVIDIA's swizzled
//                 blocked layout (swizzled_scale_index in block_scale.cuh)
//   weight_scale_2  F32 scalar
// value = e2m1(nibble) * e4m3(scale byte) * scaleFactor * globalScale, multiplied left to right so the BF16
// variant comes out bit-identical to the host reference rather than one ulp away.
//
// The codecs are decoded by hand (block_scale.cuh): __nv_cvt_fp8_to_halfraw needs SM 8.9 and this must also run on the 3060.

#include <cuda_fp16.h>

#include "block_scale.cuh"

#define NVFP4_GROUP_BYTES 8u   // 16 elements per block scale, 2 elements per packed byte

// One thread per packed byte -> two output elements. The grid is 2-D (x over the row's packed bytes, y over rows
// with a stride loop) purely to keep the per-thread integer division out of a memory-bound kernel.
#define NVFP4_DEQUANT_BODY(STORE)                                                                          \
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)                                       \
    {                                                                                                       \
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;                                           \
        if (col >= halfCols) continue;                                                                      \
        unsigned int scaleByte = blockScale[swizzled_scale_index(row, col / NVFP4_GROUP_BYTES, paddedCols)]; \
        float scale = nvfp4_e4m3_decode(scaleByte) * scaleFactor * globalScale;                             \
        unsigned int packed = weight[(unsigned long long)row * halfCols + col];                             \
        unsigned long long o = ((unsigned long long)row * halfCols + col) * 2ull;                           \
        STORE(o, nvfp4_e2m1_decode((packed >> 4) & 0xFu) * scale,                                           \
                 nvfp4_e2m1_decode(packed & 0xFu) * scale);                                                 \
    }

#define NVFP4_STORE_F16(O, EVEN, ODD)  out[(O)] = __float2half(EVEN); out[(O) + 1] = __float2half(ODD)
// Truncating F32->BF16, matching Tensor.CastTo (and therefore Nvfp4ResidentCodec) bit for bit.
#define NVFP4_STORE_BF16(O, EVEN, ODD)                                     \
    out[(O)] = (unsigned short)(__float_as_uint(EVEN) >> 16);              \
    out[(O) + 1] = (unsigned short)(__float_as_uint(ODD) >> 16)

extern "C" __global__ void dequant_nvfp4_to_f16(
    const unsigned char* __restrict__ weight, const unsigned char* __restrict__ blockScale,
    __half* __restrict__ out,
    unsigned int rows, unsigned int halfCols, unsigned int paddedCols,
    float scaleFactor, float globalScale)
{
    NVFP4_DEQUANT_BODY(NVFP4_STORE_F16)
}

extern "C" __global__ void dequant_nvfp4_to_bf16(
    const unsigned char* __restrict__ weight, const unsigned char* __restrict__ blockScale,
    unsigned short* __restrict__ out,
    unsigned int rows, unsigned int halfCols, unsigned int paddedCols,
    float scaleFactor, float globalScale)
{
    NVFP4_DEQUANT_BODY(NVFP4_STORE_BF16)
}
