// glu: a gated linear unit over the two halves of each input row (IBackend.Glu). ACT picks the gate function (0 SiLU, 1 GELU-tanh,
// 2 GELU-erf, 3 SiTU, whose value side is soft-capped to 25·tanh(u/25)); GATE_FIRST says whether the gate half leads
// ([gate | value]) or trails ([value | gate]). The sigmoid takes the sign-aware form so no tail overflows. One invocation
// per output element; USE_FP16 picks the activation dtype. Matches GluReference up to the math library.
//
// Bindings: 0=in [rows, 2*inner], 1=out [rows, inner]
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
layout(constant_id = 3) const uint ACT = 0u;
layout(constant_id = 4) const bool GATE_FIRST = true;

layout(set = 0, binding = 0) readonly  buffer In_  { DTYPE inp[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { DTYPE outp[]; };

layout(push_constant) uniform Push {
    uint inner;
    uint total;      // rows * inner
} pc;

float sigmoid_safe(float x) { return x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); }

// Abramowitz & Stegun 7.1.26, max error 1.5e-7, the form every backend's exact GELU uses.
float erf_approx(float x) {
    float s = sign(x);
    float a = abs(x);
    float k = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * exp(-a * a);
    return s * y;
}

float gate_of(float g) {
    if (ACT == 0u) return g * sigmoid_safe(g);
    if (ACT == 1u) return 0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
    if (ACT == 2u) return 0.5 * g * (1.0 + erf_approx(g * 0.7071067811865475));
    return 4.0 * tanh(g * 0.25) * sigmoid_safe(g);
}

float value_of(float u) { return ACT == 3u ? 25.0 * tanh(u / 25.0) : u; }

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint row = i / pc.inner;
    uint j = i - row * pc.inner;
    uint base = row * 2u * pc.inner;
    uint gateAt = GATE_FIRST ? base + j : base + pc.inner + j;
    uint valueAt = GATE_FIRST ? base + pc.inner + j : base + j;
    outp[i] = DTYPE(gate_of(float(inp[gateAt])) * value_of(float(inp[valueAt])));
}
