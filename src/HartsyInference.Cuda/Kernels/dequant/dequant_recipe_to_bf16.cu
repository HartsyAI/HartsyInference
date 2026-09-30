// DeepSeek-V4.1 checkpoint recipes -> BF16: routed experts (E2M1 nibbles, one F8E8M0 scale per 32 inputs) and dense/shared
// weights (E4M3 bytes, F8E8M0 scale per 32x32 block). Same decode as ModelAssets Mxfp4E8M0Codec and Fp8BlockE8M0Codec:
// value = table[q] * scale in float, then one round-to-nearest-even BF16 store. A power-of-two scale times an e2m1 or e4m3
// value is exact in BF16 except below 2^-133, so the store rounds only for scale bytes near 0. Scale byte 0 is the float
// subnormal 2^-127 (bits 0x00400000), not zero as block_scale.cuh's e8m0_decode has it, and byte 255 is NaN.
// No includes on purpose.

__device__ __forceinline__ float recipe_e8m0_scale(unsigned int b)
{
    if (b == 255u) return __uint_as_float(0x7FC00000u);
    if (b == 0u) return __uint_as_float(0x00400000u);
    return __uint_as_float(b << 23);
}

// E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; nibble 8 is +0.0 as in the reference FP4_TABLE (not -0.0).
__device__ __forceinline__ float recipe_e2m1_decode(unsigned int nibble)
{
    if (nibble == 8u) return 0.0f;
    unsigned int exponent = (nibble >> 1) & 0x3u;
    unsigned int mantissa = nibble & 0x1u;
    float magnitude = (exponent == 0u)
        ? (0.5f * mantissa)
        : (__uint_as_float((127u + exponent - 1u) << 23) * (1.0f + 0.5f * mantissa));
    return (nibble & 0x8u) ? -magnitude : magnitude;
}

// E4M3FN: bias 7, exponent 15 with mantissa 7 is NaN, subnormals man/8 * 2^-6.
__device__ __forceinline__ float recipe_e4m3_decode(unsigned int b)
{
    unsigned int exponent = (b >> 3) & 0xFu;
    unsigned int mantissa = b & 0x7u;
    float magnitude;
    if (exponent == 15u && mantissa == 7u) return __uint_as_float(0x7FC00000u);
    if (exponent == 0u) magnitude = 0.015625f * (mantissa * 0.125f);
    else magnitude = __uint_as_float((127u + exponent - 7u) << 23) * (1.0f + mantissa * 0.125f);
    return (b & 0x80u) ? -magnitude : magnitude;
}

__device__ __forceinline__ unsigned short recipe_float_to_bf16(float v)
{
    unsigned int bits = __float_as_uint(v);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;
    return (unsigned short)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// One thread per packed byte: columns 2j (low nibble) and 2j+1 (high nibble). blockCols is even, so both share a scale.
extern "C" __global__ void dequant_mxfp4_e8m0_to_bf16(
    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale, unsigned short* __restrict__ out,
    unsigned int rows, unsigned int packedCols, unsigned long long scaleStride, unsigned int scaleColOffset,
    unsigned int blockRows, unsigned int blockCols)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= packedCols) return;
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        unsigned long long scaleIndex = (unsigned long long)(row / blockRows) * scaleStride + scaleColOffset + (2u * j) / blockCols;
        float s = recipe_e8m0_scale(scale[scaleIndex]);
        unsigned int b = packed[(unsigned long long)row * packedCols + j];
        unsigned long long o = ((unsigned long long)row * packedCols + j) * 2ull;
        out[o] = recipe_float_to_bf16(recipe_e2m1_decode(b & 0xFu) * s);
        out[o + 1] = recipe_float_to_bf16(recipe_e2m1_decode(b >> 4) * s);
    }
}

extern "C" __global__ void dequant_fp8_block_e8m0_to_bf16(
    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale, unsigned short* __restrict__ out,
    unsigned int rows, unsigned int cols, unsigned long long scaleStride, unsigned int scaleColOffset,
    unsigned int blockRows, unsigned int blockCols)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        unsigned long long scaleIndex = (unsigned long long)(row / blockRows) * scaleStride + scaleColOffset + col / blockCols;
        unsigned long long i = (unsigned long long)row * cols + col;
        out[i] = recipe_float_to_bf16(recipe_e4m3_decode(packed[i]) * recipe_e8m0_scale(scale[scaleIndex]));
    }
}
