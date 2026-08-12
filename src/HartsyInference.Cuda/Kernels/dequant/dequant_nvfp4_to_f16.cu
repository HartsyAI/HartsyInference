// NVFP4 (ComfyUI `nvfp4`) packed weight -> F16/BF16, so a checkpoint quantized this way stays RESIDENT at
// 0.5 byte/param and is dequantized transiently per GEMM instead of being unpacked to 16-bit at load. That is a
// VRAM win and nothing else: neither SM 8.6 nor SM 8.9 has FP4 tensor cores, so the GEMM still runs in F16/BF16
// off the dequantized copy this kernel writes. The saving is what makes the official 18.72 GB LTX-2.5 distilled
// DiT (42 GB once unpacked) fit a 24 GB card at all.
//
// Layout, matching HartsyInference.Core.Tensors.Nvfp4ResidentCodec:
//   weight        U8 [n, k/2], two E2M1 nibbles per byte, HIGH nibble = even element (opposite of MXFP4)
//   weight_scale  E4M3 [paddedRows, paddedCols], one scale per 16 input elements, in NVIDIA's swizzled
//                 blocked layout (the permutation restated in swizzled_scale_index below)
//   weight_scale_2  F32 scalar
// value = e2m1(nibble) * e4m3(scale byte) * scaleFactor * globalScale, multiplied left to right so the BF16
// variant comes out bit-identical to the host reference rather than one ulp away.
//
// E4M3 is decoded by hand: __nv_cvt_fp8_to_halfraw needs SM 8.9 and this must also run on the 3060 (SM 8.6).
// Following this repo's E4M3FN convention there is no NaN encoding — 0x7F is the maximum magnitude 480.

#include <cuda_fp16.h>

#define NVFP4_GROUP_BYTES 8u   // 16 elements per block scale, 2 elements per packed byte

__device__ __forceinline__ float nvfp4_e4m3_decode(unsigned int b)
{
    unsigned int exponent = (b >> 3) & 0xFu;
    unsigned int mantissa = b & 0x7u;
    float magnitude;
    if (exponent == 0u)
    {
        // Subnormal: 2^-6 * (mant/8). Exact in float, so no rounding can separate this from the host table.
        magnitude = 0.015625f * (mantissa * 0.125f);
    }
    else
    {
        float power = __uint_as_float((127u + exponent - 7u) << 23);
        magnitude = power * (1.0f + mantissa * 0.125f);
    }
    return (b & 0x80u) ? -magnitude : magnitude;
}

__device__ __forceinline__ float nvfp4_e2m1_decode(unsigned int nibble)
{
    unsigned int exponent = (nibble >> 1) & 0x3u;
    unsigned int mantissa = nibble & 0x1u;
    float magnitude = (exponent == 0u)
        ? (0.5f * mantissa)
        : (__uint_as_float((127u + exponent - 1u) << 23) * (1.0f + 0.5f * mantissa));
    return (nibble & 0x8u) ? -magnitude : magnitude;
}

__device__ __forceinline__ unsigned long long swizzled_scale_index(
    unsigned int row, unsigned int blockColumn, unsigned int paddedCols)
{
    unsigned int ncb = paddedCols >> 2;
    unsigned int rb = row >> 7;
    unsigned int r128 = row & 127u;
    unsigned int a = r128 >> 5;
    unsigned int b = r128 & 31u;
    unsigned int cb = blockColumn >> 2;
    unsigned int d = blockColumn & 3u;
    unsigned long long g = (unsigned long long)rb * ncb + cb;
    return (g * 32ull + b) * 16ull + a * 4u + d;
}

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
