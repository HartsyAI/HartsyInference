// DeepSeek-V4.1 checkpoint recipes -> BF16 (plus the NVIDIA NVFP4 and MLX affine derivative formats): routed experts (E2M1 nibbles, one F8E8M0 scale per 32 inputs) and dense/shared
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

// ModelOpt NVFP4: E2M1 pairs (low nibble = even column), one E4M3 scale per blockCols inputs and a scalar F32 global scale.
// The two scales multiply first in F32, then the e2m1 value, as ModelOptNvfp4Codec does; __fmul_rn keeps the products unfused.
extern "C" __global__ void dequant_nvfp4_modelopt_to_bf16(
    const unsigned char* __restrict__ packed, const unsigned char* __restrict__ scale, unsigned short* __restrict__ out,
    unsigned int rows, unsigned int packedCols, unsigned long long scaleStride, unsigned int scaleColOffset,
    unsigned int blockRows, unsigned int blockCols, const float* __restrict__ globalScale)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= packedCols) return;
    float g = globalScale[0];
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        unsigned long long scaleIndex = (unsigned long long)(row / blockRows) * scaleStride + scaleColOffset + (2u * j) / blockCols;
        float s = __fmul_rn(recipe_e4m3_decode(scale[scaleIndex]), g);
        unsigned int b = packed[(unsigned long long)row * packedCols + j];
        unsigned long long o = ((unsigned long long)row * packedCols + j) * 2ull;
        out[o] = recipe_float_to_bf16(__fmul_rn(recipe_e2m1_decode(b & 0xFu), s));
        out[o + 1] = recipe_float_to_bf16(__fmul_rn(recipe_e2m1_decode(b >> 4), s));
    }
}

// MLX affine: q * scale + bias per group, q a 4- or 8-bit unsigned field (lowest field first). The product rounds to F32
// before the bias add, matching the host AffineIntCodec and mx.dequantize; __fmul_rn/__fadd_rn stop nvcc fusing them.
extern "C" __global__ void dequant_affine_to_bf16(
    const unsigned char* __restrict__ packed, const float* __restrict__ scale, const float* __restrict__ bias,
    unsigned short* __restrict__ out, unsigned int rows, unsigned int cols, unsigned long long scaleStride,
    unsigned int scaleColOffset, unsigned int blockRows, unsigned int group, unsigned int bits)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    for (unsigned int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        unsigned long long groupIndex = (unsigned long long)(row / blockRows) * scaleStride + scaleColOffset + col / group;
        unsigned int q;
        if (bits == 4u)
        {
            unsigned int b = packed[(unsigned long long)row * (cols / 2u) + (col >> 1)];
            q = (col & 1u) ? (b >> 4) : (b & 0xFu);
        }
        else
        {
            q = packed[(unsigned long long)row * cols + col];
        }
        float product = __fmul_rn((float)q, scale[groupIndex]);
        out[(unsigned long long)row * cols + col] = recipe_float_to_bf16(__fadd_rn(product, bias[groupIndex]));
    }
}

// EXL3 (exllamav3) 2-bit MCG trellis -> BF16, the twin of ModelAssets Exl3Codec. One block per 128x128 (output, input) Hadamard block, 256 threads:
// each thread decodes one 16x16 tile (64 bytes, a tail-biting 2-bit trellis read through a 16-bit window, MCG-hashed to two fp16 halves summed
// and rounded to fp16), the block then runs the in-axis butterfly, scales by 0.0883883.. and suh, the out-axis butterfly, scales by 0.0883883.. and
// svh, and stores one RNE BF16. The operation order and every F32 rounding match the host codec exactly (explicit __fmul_rn: no FMA contraction),
// so the BF16 output equals the host F32 result rounded once. Shared tile is [out][in] padded to 129 floats per row.
__device__ __forceinline__ float exl3_half_to_float(unsigned int bits)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)bits));
    return f;
}

__device__ __forceinline__ float exl3_round_half(float v)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return exl3_half_to_float(h);
}

// Decodes positions [first, first + 64) of one tile (16 little-endian words) into smem: position p of the stored order lands at (input it*16+r, output ot*16+c).
__device__ __forceinline__ void exl3_decode_tile_part(const unsigned int* __restrict__ words, float* s, unsigned int it, unsigned int ot, int first)
{
    for (int p = first; p < first + 64; p++)
    {
        const int start = (p * 2 + 2 - 16 + 512) & 511;
        const int wi = start >> 5, o = start & 31;
        const unsigned long long two = ((unsigned long long)words[wi] << 32) | (unsigned long long)words[(wi + 1) & 15];
        unsigned int state = (unsigned int)((two >> (48 - o)) & 0xFFFFull);
        unsigned int x = state * 0xCBAC1FEDu;
        x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
        float sum = exl3_half_to_float(x & 0xFFFFu) + exl3_half_to_float(x >> 16);
        const int lane = p >> 3, e = p & 7;
        const int r = (lane & 3) * 2 + (e & 1) + ((e >> 1) & 1) * 8;
        const int c = (lane >> 2) + (e >> 2) * 8;
        s[(ot * 16 + c) * 129 + it * 16 + r] = exl3_round_half(sum);
    }
}

// trellis: whole [in/16, out/16, 32] int16 tensor. outTiles: out/16 (the tile-row pitch). firstOutBlock: window start in 128-row blocks. cols: in.
extern "C" __global__ void dequant_exl3_2bit_to_bf16(
    const unsigned int* __restrict__ trellis, const unsigned short* __restrict__ suh, const unsigned short* __restrict__ svh,
    unsigned short* __restrict__ out, unsigned int outTiles, unsigned int firstOutBlock, unsigned int cols)
{
    extern __shared__ float exl3_smem[];
    float* s = exl3_smem;
    const unsigned int ob = firstOutBlock + blockIdx.x, ib = blockIdx.y, t = threadIdx.x;

    // A 128x128 block is 8x8 tiles = 64 tiles; four threads share a tile, 64 positions each.
    {
        const unsigned int tile = t >> 2, quarter = t & 3u;
        const unsigned int it = tile >> 3, ot = tile & 7u;
        const unsigned long long tileIndex = (unsigned long long)(ib * 8u + it) * outTiles + ob * 8u + ot;
        exl3_decode_tile_part(trellis + tileIndex * 16ull, s, it, ot, (int)quarter * 64);
    }
    __syncthreads();

    // In-axis butterfly per output row over the contiguous input axis: h = 1..64, (a + b, a - b).
    for (int h = 1; h < 128; h <<= 1)
    {
        for (unsigned int k = t; k < 8192u; k += 256u)
        {
            const unsigned int row = k >> 6, j = k & 63u;
            const unsigned int lo = ((j & ~(unsigned int)(h - 1)) << 1) | (j & (unsigned int)(h - 1));
            float* p = s + row * 129u + lo;
            const float a = p[0], b = p[h];
            p[0] = a + b;
            p[h] = a - b;
        }
        __syncthreads();
    }
    for (unsigned int k = t; k < 16384u; k += 256u)
    {
        const unsigned int row = k >> 7, il = k & 127u;
        const float v = __fmul_rn(s[row * 129u + il], 0.08838834764831845f);
        s[row * 129u + il] = __fmul_rn(v, exl3_half_to_float(suh[ib * 128u + il]));
    }
    __syncthreads();

    // Out-axis butterfly per input column at fixed il (stride 129 * h).
    for (int h = 1; h < 128; h <<= 1)
    {
        for (unsigned int k = t; k < 8192u; k += 256u)
        {
            const unsigned int il = k & 127u, j = k >> 7;
            const unsigned int lo = ((j & ~(unsigned int)(h - 1)) << 1) | (j & (unsigned int)(h - 1));
            float* p = s + lo * 129u + il;
            const float a = p[0], b = p[h * 129];
            p[0] = a + b;
            p[h * 129] = a - b;
        }
        __syncthreads();
    }
    for (unsigned int k = t; k < 16384u; k += 256u)
    {
        const unsigned int ol = k >> 7, il = k & 127u;
        const float v = __fmul_rn(s[ol * 129u + il], 0.08838834764831845f);
        const float r = __fmul_rn(v, exl3_half_to_float(svh[ob * 128u + ol]));
        out[((unsigned long long)(blockIdx.x * 128u + ol)) * cols + ib * 128u + il] = recipe_float_to_bf16(r);
    }
}
