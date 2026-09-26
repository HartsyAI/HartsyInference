// x -> E4M3 bytes at x / scale, four per word in element order, the per-tensor activation form matmul_fp8_coopmat reads.
// The scale is pc.staticScale when non-zero (a checkpoint's .input_scale), else scale[pc.scaleIndex] written by fp8_absmax.
// The conversion is block_scale.cuh's f32_to_e4m3 bit for bit: round to nearest even, |v| >= 464 saturates to 448, NaN
// stays NaN, below 2^-10 flushes to signed zero. It needs no float8 feature, so it runs on any device.
#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#else
#define DTYPE float
#endif

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_     { DTYPE x[]; };
layout(set = 0, binding = 1) writeonly buffer Q_     { uint q[]; };
layout(set = 0, binding = 2) readonly  buffer Scale_ { float scale[]; };

layout(push_constant) uniform Push {
    uint words;          // n / 4
    float staticScale;   // 0 = read the device scale
    uint scaleIndex;
} pc;

// Correctly rounded a / b, CUDA's div.rn: the driver's quotient may be an ulp off, and one exact-FMA residual step settles it.
float divRn(float a, float b) {
    precise float q = a / b;
    precise float r = fma(-q, b, a);
    return fma(r, 1.0 / b, q);
}

uint toE4M3(float f) {
    uint sign = (floatBitsToUint(f) >> 24) & 0x80u;
    float a = abs(f);
    if (isnan(a)) return sign | 0x7Fu;
    if (a >= 464.0) return sign | 0x7Eu;
    if (a < 0.0009765625) return sign;
    int e;
    float m = frexp(a, e);
    if (e - 1 >= -6) {
        int qm = int(roundEven(m * 16.0));
        if (qm == 16) { qm = 8; e += 1; }
        int expField = (e - 1) + 7;
        if (expField >= 16) return sign | 0x7Eu;
        return sign | uint(expField << 3) | uint(qm - 8);
    }
    int qs = int(roundEven(a * 512.0));
    if (qs >= 8) return sign | 0x8u;
    return sign | uint(qs);
}

void main() {
    uint w = gl_GlobalInvocationID.x;
    if (w >= pc.words) return;
    float rs = divRn(1.0, pc.staticScale != 0.0 ? pc.staticScale : scale[pc.scaleIndex]);
    uint e = w * 4u;
    q[w] = toE4M3(float(x[e]) * rs) | (toE4M3(float(x[e + 1u]) * rs) << 8)
         | (toE4M3(float(x[e + 2u]) * rs) << 16) | (toE4M3(float(x[e + 3u]) * rs) << 24);
}
