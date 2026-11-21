// exp2Fixed: 2^x computed the same way on every device, for shaders that must match a host reference bit for bit. A
// device's own exp2() is a driver's approximation and differs between drivers; this one is made of a floor, correctly
// rounded multiplies and adds, and an exact scaling by a power of two. Its twin on the host, operation for operation,
// is Int8AttentionExp2.Evaluate (HartsyInference.Core), and the two must change together.
//
// x = k + r with k the nearest integer; 2^r is the degree-7 Taylor polynomial of e^(r ln 2) by Horner's rule, within
// about one unit in the last place of the true value; the result is that times 2^k. Above 127 it is +infinity and
// below -125 exactly zero, so no result is subnormal. A NaN gives a NaN.
#ifndef EXP2_FIXED_GLSL
#define EXP2_FIXED_GLSL

float exp2Fixed(float x) {
    if (isnan(x)) return x;
    if (x > 127.0) return uintBitsToFloat(0x7F800000u);
    if (x < -125.0) return 0.0;
    precise float k = floor(x + 0.5);
    precise float r = x - k;
    // ln(2)^n / n!, n = 7 .. 1, as the float nearest each.
    precise float p = uintBitsToFloat(0x377FE5FEu);
    p = p * r + uintBitsToFloat(0x39218489u);
    p = p * r + uintBitsToFloat(0x3AAEC3FFu);
    p = p * r + uintBitsToFloat(0x3C1D955Bu);
    p = p * r + uintBitsToFloat(0x3D635847u);
    p = p * r + uintBitsToFloat(0x3E75FDF0u);
    p = p * r + uintBitsToFloat(0x3F317218u);
    p = p * r + 1.0;
    float scale = uintBitsToFloat(uint(int(k) + 127) << 23);
    precise float y = p * scale;
    return y;
}

#endif
