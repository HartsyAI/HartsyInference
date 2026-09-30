// Scalar codecs for the DeepSeek-V4.1 latent formats, mirroring LatentCodec (Core) bit for bit. Encodings: 0 F32, 1 fp8
// e4m3 with a ue8m0 scale per 32, 2 fp4 e2m1 with an e4m3 scale per 16, 3 fp4 e2m1 with a ue8m0 scale per 32. Code and scale
// bytes live in uint SSBOs (no 8-bit storage feature needed); a source's byte sizes must be multiples of 4.

const float E2M1_TABLE[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
// Bit patterns kept as functions: GLSL does not fold uintBitsToFloat into a constant.
float fp8MaxInv() { return uintBitsToFloat(0x3b124925u); }   // 1f / 448f
float fp4MaxInv() { return uintBitsToFloat(0x3e2aaaabu); }   // 1f / 6f
float fp4MinUe8m0() { return uintBitsToFloat(0x01c00000u); } // 6f * MinNormal

uint latentGroupSize(uint enc) {
    return enc == 2u ? 16u : 32u;
}

float decodeE4M3(uint code) {
    uint ex = (code >> 3) & 0xFu, man = code & 7u;
    if (ex == 0xFu && man == 7u) return uintBitsToFloat(0x7FC00000u);
    uint signBit = (code & 0x80u) << 24;
    if (ex == 0u) return uintBitsToFloat(floatBitsToUint(float(man) * 0.001953125) | signBit);
    return uintBitsToFloat(((ex + 120u) << 23) | (man << 20) | signBit);
}

float decodeUe8m0(uint code) {
    return uintBitsToFloat(code == 0u ? 0x00400000u : code << 23);
}

float decodeE2M1(uint code) {
    float v = E2M1_TABLE[code & 7u];
    return (code & 8u) != 0u ? -v : v;
}

// Round-to-nearest-even e4m3fn encode with the sign kept for zeros; past 464 it is NaN (0x7F), as the cast is.
uint encodeE4M3(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = (bits >> 24) & 0x80u;
    float mag = abs(value);
    uint code;
    if (mag < 0.015625) {
        code = uint(roundEven(mag * 512.0));
    } else {
        uint m = bits & 0x7FFFFFFFu;
        uint rounded = (m + 0x7FFFFu + ((m >> 20) & 1u)) >> 20;
        uint c = rounded - 960u;
        code = c > 0x7Eu ? 0x7Fu : c;
    }
    return sign | code;
}

// E2M1 code of a value already clamped to +-6: nearest, ties to the even code; -0 keeps its sign bit, NaN rounds to +0.
uint encodeE2M1(float value) {
    float mag = abs(value);
    uint code = 0u;
    for (int i = 0; i < 7; i++) {
        float mid = (E2M1_TABLE[i] + E2M1_TABLE[i + 1]) * 0.5;
        if (mag > mid) code++;
        else if (mag == mid && (i & 1) == 1) code++;
    }
    bool negative = value < 0.0 || (value == 0.0 && (floatBitsToUint(value) >> 31) != 0u);
    return code | (negative ? 8u : 0u);
}

// Power-of-two ceiling of amax * maxInv read from the fp32 exponent; returns the e8m0 byte.
uint ue8m0Byte(float amax, float maxInv) {
    precise float scaled = amax * maxInv;
    int bits = floatBitsToInt(scaled);
    int ex = (bits >> 23) & 0xFF;
    int log2Ceil = ex - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
    return uint(log2Ceil + 127) & 0xFFu;
}

#ifdef LATENT_ENCODE
// (define LATENT_ENCODE and include div_rn.glsl first)
// Scale byte for a group whose largest magnitude is amax, and the decoded scale it stands for.
uint latentScaleByte(uint enc, float amax, out float scale) {
    uint sb;
    if (enc == 1u) {
        sb = ue8m0Byte(max(amax, 1e-4), fp8MaxInv());
        scale = decodeUe8m0(sb);
    } else if (enc == 2u) {
        sb = encodeE4M3(divRn(max(amax, 0.01171875), 6.0));
        scale = decodeE4M3(sb);
    } else {
        sb = ue8m0Byte(max(amax, fp4MinUe8m0()), fp4MaxInv());
        scale = decodeUe8m0(sb);
    }
    return sb;
}

// Code (fp8 byte or fp4 nibble) of one element under the group's decoded scale.
uint latentCode(uint enc, float v, float scale) {
    float q = divRn(v, scale);
    // Math.Clamp propagates NaN (a NaN scale from an overflowed e4m3 group); GLSL min/max would return the bound.
    bool nan = q != q;
    if (enc == 1u) return encodeE4M3(nan ? q : min(max(q, -448.0), 448.0));
    return encodeE2M1(nan ? q : min(max(q, -6.0), 6.0));
}

#endif

// Decoded value of an element's code before the scale is applied.
float latentDecodeCode(uint enc, uint code) {
    return enc == 1u ? decodeE4M3(code) : decodeE2M1(code);
}

// Defines float NAME(enc, row, dim, col): element col of row in the source held by the uint SSBOs CODES and SCALES.
#define LATENT_DECODE_FN(NAME, CODES, SCALES)                                                                        \
float NAME(uint enc, uint row, uint dim, uint col) {                                                                 \
    if (enc == 0u) return uintBitsToFloat(CODES[row * dim + col]);                                                   \
    uint group = latentGroupSize(enc);                                                                               \
    uint sIdx = row * (dim / group) + col / group;                                                                   \
    uint sByte = (SCALES[sIdx >> 2] >> ((sIdx & 3u) * 8u)) & 0xFFu;                                                  \
    float scale = enc == 2u ? decodeE4M3(sByte) : decodeUe8m0(sByte);                                                \
    if (enc == 1u) {                                                                                                 \
        uint cIdx = row * dim + col;                                                                                 \
        return decodeE4M3((CODES[cIdx >> 2] >> ((cIdx & 3u) * 8u)) & 0xFFu) * scale;                                 \
    }                                                                                                                \
    uint pIdx = row * (dim >> 1) + (col >> 1);                                                                       \
    uint packed = (CODES[pIdx >> 2] >> ((pIdx & 3u) * 8u)) & 0xFFu;                                                  \
    return decodeE2M1((col & 1u) == 0u ? packed & 0xFu : packed >> 4) * scale;                                       \
}
