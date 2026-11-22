// Correctly rounded fp32 a*b + c with one rounding, the result of MathF.FusedMultiplyAdd and CUDA's fmaf, for shaders
// that must match a host reference bit for bit. Vulkan need not fuse fma(), but OpFMul and OpFAdd are correctly rounded,
// so the product is taken exactly (Dekker) and added to c with the low part rounded to odd (Boldo and Melquiond, 2008),
// which keeps the final add from rounding twice.
//
// Exact when all of these hold:
//   a and b are zero or normal floats (at least 2^-126 in magnitude), and at most 2^100;
//   a*b is zero, or |a*b| is within [2^-124, 2^120];
//   c is zero or a normal float, and at most 2^120;
//   the result is zero or a normal float.
// Small operands, products and addends are moved up by exact powers of two before the work, so that every split
// part, partial product and error term is a whole multiple of something normal and none is subnormal (a device may
// read a subnormal as zero):
//   a product below 2^-40: the smaller operand and the addend are scaled by 2^72, and the result scaled back;
//   otherwise an operand below 2^-60: it is scaled by 2^72 and the other by 2^-72, which leaves the product;
//   otherwise an addend below 2^-100: it is so far under the product's last place that only its sign can matter,
//   and 2^-100 with that sign stands in for it.
// Where an operand, the product or the addend is outside those ranges, the result is the plain a*b + c of the device.
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
    precise float plain = p + c;
    if (a == 0.0 || b == 0.0 || c == 0.0 || isnan(p) || isinf(p) || isnan(c) || isinf(c)) return plain;
    // 2^-100, 2^100 and 2^120; 2^-124 for the product; 2^-126, the smallest normal float.
    const float LOW = 7.88860905e-31, HIGH = 1.2676506e30, TOP = 1.329228e36, PLOW = 4.70197740e-38;
    const float NORMAL = 1.17549435e-38;
    float aa = abs(a), ab = abs(b), ap = abs(p), ac = abs(c);
    if (min(aa, ab) < NORMAL || max(aa, ab) > HIGH || ap < PLOW || ap > TOP || ac < NORMAL || ac > TOP) return plain;
    float back = 1.0;
    // A product below 2^-40: against an addend of 2^28 or more it is far under a quarter of the addend's last place,
    // and the fused result is the addend, as the plain one is. Otherwise scale by 2^72, which is exact here.
    if (ap < 9.09494702e-13) {
        if (ac >= 268435456.0) return plain;
        if (aa < ab) a *= 4.72236648e21;
        else b *= 4.72236648e21;
        c *= 4.72236648e21;
        back = 2.11758237e-22;
    } else {
        // The product is at least 2^-40, so an operand below 2^-60 has a partner of at least 2^20.
        if (aa < 8.67361738e-19) { a *= 4.72236648e21; b *= 2.11758237e-22; }
        else if (ab < 8.67361738e-19) { b *= 4.72236648e21; a *= 2.11758237e-22; }
        if (ac < LOW) c = c < 0.0 ? -LOW : LOW;
    }
    precise float ps = a * b;
    // ps + e = a * b exactly.
    precise float ca = 4097.0 * a, ah = ca - (ca - a), al = a - ah;
    precise float cb = 4097.0 * b, bh = cb - (cb - b), bl = b - bh;
    precise float e = ((ah * bh - ps) + ah * bl + al * bh) + al * bl;
    // th + tl = c + ps exactly; v is tl + e rounded to odd: where that sum is inexact and its rounding has an even last
    // bit, the neighbour on the side of the error is the odd one.
    float th, tl, v, ve;
    fmaTwoSum(c, ps, th, tl);
    fmaTwoSum(tl, e, v, ve);
    if (ve != 0.0 && (floatBitsToUint(v) & 1u) == 0u) {
        bool away = (ve > 0.0) == (v > 0.0);
        v = uintBitsToFloat(floatBitsToUint(v) + (away ? 1u : 0xFFFFFFFFu));
    }
    precise float r = th + v;
    precise float scaled = r * back;
    return scaled;
}

#endif
