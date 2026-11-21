// Correctly rounded fp32 a*b + c with one rounding, the result of MathF.FusedMultiplyAdd and CUDA's fmaf, for shaders
// that must match a host reference bit for bit. Vulkan need not fuse fma(), but OpFMul and OpFAdd are correctly rounded,
// so the product is taken exactly (Dekker) and added to c with the low part rounded to odd (Boldo and Melquiond, 2008),
// which keeps the final add from rounding twice.
//
// Exact while nothing overflows and no intermediate is subnormal: |a|, |b| within [2^-40, 2^40] or zero, and c zero or
// within [2^-80, 2^80]. Outside that the result is the plain a*b + c of the device.
#ifndef FMA_RN_GLSL
#define FMA_RN_GLSL

// s + t = a + b exactly, s the rounded sum (Knuth).
void fmaTwoSum(float a, float b, out float s, out float t) {
    precise float sum = a + b;
    precise float bv = sum - a;
    precise float av = sum - bv;
    precise float br = b - bv;
    precise float ar = a - av;
    precise float err = ar + br;
    s = sum;
    t = err;
}

float fmaRn(float a, float b, float c) {
    precise float p = a * b;
    if (a == 0.0 || b == 0.0 || c == 0.0 || isnan(p) || isinf(p) || isnan(c) || isinf(c)) {
        precise float plain = p + c;
        return plain;
    }
    // p + e = a * b exactly.
    precise float ca = 4097.0 * a, ah = ca - (ca - a), al = a - ah;
    precise float cb = 4097.0 * b, bh = cb - (cb - b), bl = b - bh;
    precise float e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    // th + tl = c + p exactly; v is tl + e rounded to odd: where that sum is inexact and its rounding has an even last
    // bit, the neighbour on the side of the error is the odd one.
    float th, tl, v, ve;
    fmaTwoSum(c, p, th, tl);
    fmaTwoSum(tl, e, v, ve);
    if (ve != 0.0 && (floatBitsToUint(v) & 1u) == 0u) {
        bool away = (ve > 0.0) == (v > 0.0);
        v = uintBitsToFloat(floatBitsToUint(v) + (away ? 1u : 0xFFFFFFFFu));
    }
    precise float r = th + v;
    return r;
}

#endif
