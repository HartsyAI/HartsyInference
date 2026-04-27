// elementwise: unary and binary elementwise ops on contiguous tensors.
// Op selected by spec const ELEMENTWISE_OP:
//   0=add(a,b)  1=mul(a,b)  2=scale(a,scalar)  3=silu(a)  4=gelu_exact(a)
//   5=gelu_tanh(a)  6=sigmoid(a)  7=clamp(a,minVal,maxVal)
//
// All math accumulated in fp32. FP16 inputs/outputs widen to fp32 for compute,
// narrow back at write — matches the CUDA PTX convention.
//
// Bindings: 0 = a (input)
//           1 = b (input, used only when binary op)
//           2 = out (output, written to first; aliased to a for unary ops)
// Note: for unary ops, the host must bind `a` at both binding 0 and binding 2
// (or pass a separate output buffer). The kernel reads from binding 0 and writes
// to binding 2 unconditionally to keep the dispatch shape uniform.
#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#extension GL_KHR_shader_subgroup_basic : require
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
layout(constant_id = 10) const uint ELEMENTWISE_OP = 0;

layout(set = 0, binding = 0) readonly  buffer A_  { DTYPE a[];   };
layout(set = 0, binding = 1) readonly  buffer B_  { DTYPE b[];   };
layout(set = 0, binding = 2) writeonly buffer O_  { DTYPE out_[]; };

layout(push_constant) uniform Push {
    uint  count;
    float scalar;
    float minVal;
    float maxVal;
} pc;

float silu(float x) { return x / (1.0 + exp(-x)); }
// erf approximation, Abramowitz & Stegun 7.1.26 (max error 1.5e-7)
float erf_approx(float x) {
    float s = sign(x);
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}
float gelu_exact(float x) { return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865475)); }
float gelu_tanh(float x)  { return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
float sigmoid(float x)    { return 1.0 / (1.0 + exp(-x)); }

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;

    float av = TO_F32(a[i]);
    float r;
    if      (ELEMENTWISE_OP == 0u) r = av + TO_F32(b[i]);
    else if (ELEMENTWISE_OP == 1u) r = av * TO_F32(b[i]);
    else if (ELEMENTWISE_OP == 2u) r = av * pc.scalar;
    else if (ELEMENTWISE_OP == 3u) r = silu(av);
    else if (ELEMENTWISE_OP == 4u) r = gelu_exact(av);
    else if (ELEMENTWISE_OP == 5u) r = gelu_tanh(av);
    else if (ELEMENTWISE_OP == 6u) r = sigmoid(av);
    else /* 7 */                   r = clamp(av, pc.minVal, pc.maxVal);
    out_[i] = FROM_F32(r);
}
