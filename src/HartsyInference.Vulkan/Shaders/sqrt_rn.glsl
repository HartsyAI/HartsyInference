// Correctly rounded fp32 square root, built on div_rn.glsl's exact residual. Its own file so that shaders which only
// divide keep their SPIR-V: an include is compiled into every shader that names it, whether its functions are called or not.
#ifndef SQRT_RN_GLSL
#define SQRT_RN_GLSL

#include "div_rn.glsl"

// Vulkan's sqrt(a) may be a few ulp off. Take the root of the mantissa with an even exponent (so squaring a candidate
// cannot overflow or underflow), refine once from the exact residual m - s*s, then keep the nearest of the three
// candidates around it. A square root of a float is never exactly halfway between two floats, so nearest is unambiguous.
float sqrtRn(float a) {
    if (!(a > 0.0) || isinf(a)) return sqrt(a);
    int e;
    float m = frexp(a, e);              // m in [0.5, 1)
    if ((e & 1) != 0) { m = m * 2.0; e -= 1; }   // m in [0.5, 2), e even
    precise float s = sqrt(m);
    precise float step = divResidual(m, s, s) * (0.5 / s);
    s = s + step;
    uint bits = floatBitsToUint(s);
    float best = s, bestR = abs(divResidual(m, s, s));
    for (int d = -1; d <= 1; d += 2) {
        float c = uintBitsToFloat(bits + uint(d));
        float cr = abs(divResidual(m, c, c));
        if (cr < bestR) { best = c; bestR = cr; }
    }
    return ldexp(best, e / 2);
}

#endif
