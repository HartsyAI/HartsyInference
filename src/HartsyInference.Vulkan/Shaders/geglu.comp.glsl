// GEGLU: Splits last dim in half. y[..., d] = x[..., d] * gelu_tanh(x[..., d + D])
// Output is half the elements of input.
//
// CRITICAL: split along the LAST dim — not flat midpoint. See PHASE_3_DEVIATIONS #16.
//
// Bindings: 0=in (size 2*D last dim), 1=out (size D last dim)
#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#define TO_F32(x) float(x)
#define FROM_F32(x) float16_t(x)
#else
#define DTYPE float
#define TO_F32(x) (x)
#define FROM_F32(x) (x)
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
// Which GELU gates the value. The two differ by ~1e-3 at the tails, which is visible in a DiT's output.
layout(constant_id = 3) const bool USE_ERF = false;

layout(set = 0, binding = 0) readonly  buffer In_  { DTYPE inp[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { DTYPE outp[]; };

layout(push_constant) uniform Push {
    uint outerCount;     // product of all dims except last
    uint D;              // half-size of last dim
} pc;

float gelu_tanh(float x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

// Abramowitz & Stegun 7.1.26, max error 1.5e-7 — the same approximation the elementwise kernel uses, so
// GeluErf and GegluErf agree with each other and with the host reference.
float erf_approx(float x) {
    float s = sign(x);
    float a = abs(x);
    float k = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * exp(-a * a);
    return s * y;
}
float gelu_exact(float x) { return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865475)); }

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = pc.outerCount * pc.D;
    if (i >= total) return;

    uint outer = i / pc.D;
    uint d = i % pc.D;
    uint baseIn = outer * 2u * pc.D;
    float val = TO_F32(inp[baseIn + d]);
    float gate = TO_F32(inp[baseIn + pc.D + d]);
    outp[i] = FROM_F32(val * (USE_ERF ? gelu_exact(gate) : gelu_tanh(gate)));
}
