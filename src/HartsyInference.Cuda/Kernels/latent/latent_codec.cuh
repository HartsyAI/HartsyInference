// Device codecs for the latent-cache formats. Every function mirrors the scalar CPU codec (LatentCodec.cs), which
// reproduces the reference act_quant / fp4_act_quant bit for bit:
//   encoding 0 = F32                 (no scales)
//   encoding 1 = FP8 e4m3   + ue8m0 scale per 32 elements
//   encoding 2 = FP4 e2m1   + e4m3  scale per 16 elements (two elements per byte, even element in the low nibble)
//   encoding 3 = FP4 e2m1   + ue8m0 scale per 32 elements
// All arithmetic that must round like the host is IEEE (no FMA contraction, correctly rounded division).

#ifndef LATENT_CODEC_CUH
#define LATENT_CODEC_CUH

#define LC_F32 0
#define LC_FP8 1
#define LC_FP4_E4M3 2
#define LC_FP4_E8M0 3

__device__ __forceinline__ int lc_group_size(int enc)
{
    return enc == LC_F32 ? 0 : (enc == LC_FP4_E4M3 ? 16 : 32);
}

__device__ __forceinline__ float lc_decode_e4m3(unsigned int code)
{
    const int e = (code >> 3) & 0xF, m = code & 7;
    if (e == 0xF && m == 7) return __int_as_float(0x7fc00000);
    const float sign = (code & 0x80u) ? -1.0f : 1.0f;
    if (e == 0) return __fmul_rn(__fmul_rn(sign, (float)m), 0.001953125f);
    return __fmul_rn(sign, __int_as_float(((e + 120) << 23) | (m << 20)));
}

// Round-to-nearest-even e4m3fn encode; the sign of a zero is kept and anything past 464 becomes NaN (0x7F).
__device__ __forceinline__ unsigned int lc_encode_e4m3(float value)
{
    const unsigned int bits = (unsigned int)__float_as_int(value);
    const unsigned int sign = (bits >> 24) & 0x80u;
    const float mag = fabsf(value);
    unsigned int code;
    if (mag < 0.015625f)
    {
        code = (unsigned int)rintf(__fmul_rn(mag, 512.0f));
    }
    else
    {
        const unsigned int m = bits & 0x7FFFFFFFu;
        const unsigned int rounded = (m + 0x7FFFFu + ((m >> 20) & 1u)) >> 20;
        const unsigned int c = rounded - 960u;
        code = c > 0x7Eu ? 0x7Fu : c;
    }
    return sign | code;
}

__device__ __forceinline__ float lc_decode_ue8m0(unsigned int code)
{
    return __int_as_float(code == 0 ? 0x00400000 : (int)(code << 23));
}

__device__ __forceinline__ float lc_decode_e2m1(unsigned int code)
{
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = table[code & 7];
    return (code & 8) ? -v : v;
}

// Code of a value already clamped to +-6: round to nearest, ties to the even code, -0 keeps its sign bit,
// NaN (an overflowed e4m3 scale) has no sign and rounds to +0.
__device__ __forceinline__ unsigned int lc_encode_e2m1(float value)
{
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float mag = fabsf(value);
    unsigned int code = 0;
    for (int i = 0; i < 7; i++)
    {
        const float mid = __fmul_rn(__fadd_rn(table[i], table[i + 1]), 0.5f);
        if (mag > mid) code++;
        else if (mag == mid && (i & 1) == 1) code++;
    }
    const bool negative = value < 0.0f || (value == 0.0f && ((unsigned int)__float_as_int(value) >> 31) != 0);
    return code | (negative ? 8u : 0u);
}

// Power-of-two ceiling of amax * maxInv, read from the fp32 exponent; returns the e8m0 byte.
__device__ __forceinline__ unsigned int lc_ue8m0_byte(float amax, float maxInv)
{
    const int bits = __float_as_int(__fmul_rn(amax, maxInv));
    const int exp = (bits >> 23) & 0xFF;
    const int log2Ceil = exp - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
    return (unsigned int)(log2Ceil + 127) & 0xFFu;
}

__device__ __forceinline__ float lc_decode_scale(int enc, unsigned int byte)
{
    return enc == LC_FP4_E4M3 ? lc_decode_e4m3(byte) : lc_decode_ue8m0(byte);
}

// Element `col` of row `row` of a cache of `dim`-wide rows in encoding `enc`.
__device__ __forceinline__ float lc_load(int enc, const unsigned char* codes, const unsigned char* scales, long long row,
                                         int dim, int col)
{
    if (enc == LC_F32) return ((const float*)codes)[row * dim + col];
    const int group = lc_group_size(enc);
    const float scale = lc_decode_scale(enc, scales[row * (dim / group) + col / group]);
    if (enc == LC_FP8) return __fmul_rn(lc_decode_e4m3(codes[row * dim + col]), scale);
    const unsigned int packed = codes[row * (dim >> 1) + (col >> 1)];
    return __fmul_rn(lc_decode_e2m1((col & 1) == 0 ? (packed & 0xFu) : (packed >> 4)), scale);
}

// Quantizes n (<= 32) values of one scale group. Writes one code per element (fp8 byte or fp4 nibble) to `out`
// and returns the decoded scale; scaleByte receives the stored scale byte.
__device__ __forceinline__ float lc_quantize_group(int enc, const float* g, int n, unsigned char* out,
                                                   unsigned int* scaleByte)
{
    float amax = 0.0f;
    for (int i = 0; i < n; i++) amax = fmaxf(amax, fabsf(g[i]));
    float scale;
    float codeMax;
    if (enc == LC_FP8)
    {
        *scaleByte = lc_ue8m0_byte(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
        scale = lc_decode_ue8m0(*scaleByte);
        codeMax = 448.0f;
    }
    else if (enc == LC_FP4_E4M3)
    {
        *scaleByte = lc_encode_e4m3(__fdiv_rn(fmaxf(amax, __fmul_rn(6.0f, 0.001953125f)), 6.0f));
        scale = lc_decode_e4m3(*scaleByte);
        codeMax = 6.0f;
    }
    else
    {
        *scaleByte = lc_ue8m0_byte(fmaxf(amax, __fmul_rn(6.0f, __int_as_float(0x00800000))), 1.0f / 6.0f);
        scale = lc_decode_ue8m0(*scaleByte);
        codeMax = 6.0f;
    }
    for (int i = 0; i < n; i++)
    {
        const float v = __fdiv_rn(g[i], scale);
        const float c = v < -codeMax ? -codeMax : (v > codeMax ? codeMax : v);
        out[i] = (unsigned char)(enc == LC_FP8 ? lc_encode_e4m3(c) : lc_encode_e2m1(c));
    }
    return scale;
}

#endif
