// Correctly rounded fp32 division, CUDA's div.rn, shared by fp8_absmax and quant_e4m3 so scale and reciprocal agree.

// a - q*b to within one rounding. Vulkan need not fuse fma(), but OpFMul/OpFAdd are correctly rounded: Dekker's exact product.
float divResidual(float a, float b, float q) {
    precise float cq = 4097.0 * q, qh = cq - (cq - q), ql = q - qh;
    precise float cb = 4097.0 * b, bh = cb - (cb - b), bl = b - bh;
    precise float p = q * b;
    precise float e = ((qh * bh - p) + qh * bl + ql * bh) + ql * bl;
    precise float r = (a - p) - e;
    return r;
}

// Vulkan's a / b may be 2.5 ulp off: divide the mantissas (so the splitter cannot overflow), refine once, then keep the
// nearest candidate on the result's own grid, which below 2^-126 is whole multiples of 2^-149 (so a subnormal is rounded once).
float divRn(float a, float b) {
    if (a == 0.0 || isinf(a) || isnan(a) || b == 0.0 || isinf(b) || isnan(b)) return a / b;
    uint sign = (floatBitsToUint(a) ^ floatBitsToUint(b)) & 0x80000000u;
    int ea, eb;
    float ma = frexp(abs(a), ea), mb = frexp(abs(b), eb);
    int e = ea - eb;
    precise float q = ma / mb;
    precise float step = divResidual(ma, mb, q) * (1.0 / mb);
    q = q + step;
    if (e + 149 >= 24) {
        uint bits = floatBitsToUint(q);
        float best = q, bestR = abs(divResidual(ma, mb, q));
        for (int d = -1; d <= 1; d += 2) {
            float c = uintBitsToFloat(bits + uint(d));
            float cr = abs(divResidual(ma, mb, c));
            if (cr < bestR) { best = c; bestR = cr; }
        }
        return uintBitsToFloat(floatBitsToUint(ldexp(best, e)) | sign);
    }
    int s = e + 149;
    float k = floor(ldexp(q, s) + 0.5), bestK = k, bestR = 1.0 / 0.0;
    for (int d = -1; d <= 1; d++) {
        float c = k + float(d);
        if (c < 0.0) continue;
        float cr = abs(divResidual(ma, mb, ldexp(c, -s)));
        if (cr < bestR || (cr == bestR && (int(c) & 1) == 0)) { bestK = c; bestR = cr; }
    }
    return uintBitsToFloat(uint(bestK) | sign);
}
