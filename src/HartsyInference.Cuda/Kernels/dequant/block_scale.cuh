// Device helpers shared by every kernel that reads or writes a microscaled (block-scaled) operand:
// NVIDIA's blocked scale layout, the e4m3/e2m1 codecs, and the F16 bit reader. Hand-rolled bit math throughout so
// the same PTX JITs on Ampere for the unit tests; the sm_100a/sm_120a builds swap in the hardware e2m1 cvt.
// No includes on purpose: fp8_quant.cu is compiled without the CUDA headers.
//
// Blocked layout (matching HartsyInference.Core.Tensors.Nvfp4ResidentCodec.SwizzledScaleIndex): a logical
// [rows, blockCols] scale tensor is zero-padded to [128·ceil(rows/128), 4·ceil(blockCols/4)] and element (r, c) lands at
// ((g·32 + b)·16 + a·4 + d) with g = (r/128)·(paddedCols/4) + c/4, a = (r%128)/32, b = r%32, d = c%4.
#pragma once

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

// E4M3 byte -> float. This repo's E4M3FN convention has no NaN encoding — 0x7F is the maximum magnitude 480.
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

// E2M1 nibble -> float (bit 3 sign; magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6).
__device__ __forceinline__ float nvfp4_e2m1_decode(unsigned int nibble)
{
    unsigned int exponent = (nibble >> 1) & 0x3u;
    unsigned int mantissa = nibble & 0x1u;
    float magnitude = (exponent == 0u)
        ? (0.5f * mantissa)
        : (__uint_as_float((127u + exponent - 1u) << 23) * (1.0f + 0.5f * mantissa));
    return (nibble & 0x8u) ? -magnitude : magnitude;
}

// UE8M0 byte -> float: 2^(byte - 127), and 0 for byte 0 — the bit reinterpretation byte << 23.
__device__ __forceinline__ float e8m0_decode(unsigned int b) { return __uint_as_float(b << 23); }

// The UE8M0 shared exponent for a block whose |max| is amax, chosen so the largest element lands at or under e4m3's
// max normal (448 = 1.75·2^8): 2^(floor(log2 amax) - 8), as OCP MX specifies. 0 for an all-zero block. frexpf gives
// floor(log2) exactly, which log2f does not at powers of two.
__device__ __forceinline__ unsigned int e8m0_of(float amax)
{
    if (!(amax > 0.0f)) return 0u;
    int e;
    frexpf(amax, &e);                       // amax = m·2^e, m in [0.5, 1)  =>  floor(log2 amax) = e - 1
    int stored = (e - 1) - 8 + 127;
    return (unsigned int)(stored < 1 ? 1 : stored > 254 ? 254 : stored);
}

// float -> e4m3fn (bias 7, max normal 448, no inf; 0x7F/0xFF = NaN). Round-half-away-from-zero on
// the mantissa (vs the IEEE ties-to-even a hardware cvt would do) — a <=0.5-ulp difference on exact
// ties only, irrelevant for activation quantization. Values past the 448+16 rounding midpoint clamp
// to +-448 (satfinite semantics: quantization must never emit NaN).
__device__ __forceinline__ unsigned char f32_to_e4m3(float f)
{
    unsigned char sign = (unsigned char)((__float_as_uint(f) >> 24) & 0x80u);
    float a = fabsf(f);
    if (!(a == a)) return (unsigned char)(sign | 0x7Fu);       // NaN propagates as e4m3 NaN
    if (a >= 464.0f) return (unsigned char)(sign | 0x7Eu);     // clamp to max normal 448
    if (a < 0.0009765625f) return sign;                        // < 2^-10 = half of min subnormal -> +-0
    int e;
    float m = frexpf(a, &e);                                   // a = m * 2^e, m in [0.5, 1)
    if (e - 1 >= -6)                                           // normal: value = (2m) * 2^(e-1), 2m in [1,2)
    {
        int q = (int)rintf(m * 16.0f);                         // round mantissa to 4 bits (8..16)
        if (q == 16) { q = 8; e += 1; }                        // mantissa overflow -> bump exponent
        int expField = (e - 1) + 7;                            // e4m3 bias 7
        if (expField >= 16) return (unsigned char)(sign | 0x7Eu);
        return (unsigned char)(sign | (expField << 3) | (q - 8));
    }
    // subnormal: value = q * 2^-9, q in [0, 8); q==8 rolls into the smallest normal (2^-6).
    int q = (int)rintf(a * 512.0f);
    if (q >= 8) return (unsigned char)(sign | (1 << 3));
    return (unsigned char)(sign | q);
}

// float -> e2m1 nibble: round to nearest on the {0, .5, 1, 1.5, 2, 3, 4, 6} grid, ties to the even mantissa,
// |v| past 6 saturates and NaN saturates too — the semantics of the sm_100a/sm_120a cvt.rn.satfinite.e2m1x2.f32,
// so the two builds quantize identically.
__device__ __forceinline__ unsigned int f32_to_e2m1(float v)
{
    unsigned int sign = (__float_as_uint(v) >> 28) & 0x8u;
    float a = fabsf(v);
    if (!(a == a)) a = 6.0f;
    unsigned int code;
    if      (a <= 0.25f) code = 0u;
    else if (a <  0.75f) code = 1u;
    else if (a <= 1.25f) code = 2u;
    else if (a <  1.75f) code = 3u;
    else if (a <= 2.5f)  code = 4u;
    else if (a <  3.5f)  code = 5u;
    else if (a <= 5.0f)  code = 6u;
    else                 code = 7u;
    return sign | code;
}

// Two floats -> one packed byte, HIGH nibble = even element, the order dequant_nvfp4_to_f16.cu reads weights in.
__device__ __forceinline__ unsigned int e2m1x2_pack(float even, float odd)
{
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) \
    || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    unsigned short packed;   // cvt writes the first operand to the upper nibble
    asm("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(even), "f"(odd));
    return (unsigned int)(packed & 0xFFu);
#else
    return (f32_to_e2m1(even) << 4) | f32_to_e2m1(odd);
#endif
}

// __half bits -> float without cuda_fp16.h.
__device__ __forceinline__ float h16_to_f32(unsigned short h)
{
    unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    unsigned int expo = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (expo == 0u)
    {
        if (mant == 0u) { bits = sign; }                        // +-0
        else
        {
            // subnormal: normalize
            int e = -1;
            do { mant <<= 1; e++; } while ((mant & 0x400u) == 0u);
            bits = sign | ((unsigned int)(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    }
    else if (expo == 0x1Fu) { bits = sign | 0x7F800000u | (mant << 13); }   // inf/NaN
    else { bits = sign | ((expo - 15u + 127u) << 23) | (mant << 13); }
    return __uint_as_float(bits);
}
