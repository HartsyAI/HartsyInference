// group_scale: every group of a row scaled by its own gate (IBackend.GroupScale): out[i] = f(gate(row, group)) · in[i], or out[i] +=
// that with a zero gate skipped (ACCUMULATE). FUNCTION 0 takes the stored gate, 1 takes 2·sigmoid. The gate of row r, group g sits
// at gateOffset + r·gateStride + g; GATE_FP16 reads F16 gates, else F32. The output may be the input. One invocation per element.
// Matches GroupScaleReference up to the math library.
//
// Bindings: 0=in, 1=gates, 2=out
#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif
#ifndef GATE_FP16
#define GATE_FP16 0
#endif
#if USE_FP16 == 1 || GATE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#endif
#if USE_FP16 == 1
#define DTYPE float16_t
#else
#define DTYPE float
#endif
#if GATE_FP16 == 1
#define GTYPE float16_t
#else
#define GTYPE float
#endif

layout(local_size_x_id = 0) in;
layout(constant_id = 3) const uint FUNCTION = 0u;
layout(constant_id = 4) const bool ACCUMULATE = false;

layout(set = 0, binding = 0) readonly buffer In_    { DTYPE inp[]; };
layout(set = 0, binding = 1) readonly buffer Gates_ { GTYPE gates[]; };
layout(set = 0, binding = 2)          buffer Out_   { DTYPE outp[]; };

layout(push_constant) uniform Push {
    uint total;
    uint width;
    uint groupWidth;
    uint gateStride;
    uint gateOffset;
} pc;

float sigmoid_safe(float x) { return x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); }

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint row = i / pc.width;
    uint group = (i % pc.width) / pc.groupWidth;
    float gate = float(gates[pc.gateOffset + row * pc.gateStride + group]);
    float factor = FUNCTION == 1u ? 2.0 * sigmoid_safe(gate) : gate;
    if (ACCUMULATE) {
        if (factor != 0.0) outp[i] = DTYPE(float(outp[i]) + factor * float(inp[i]));
    } else {
        outp[i] = DTYPE(factor * float(inp[i]));
    }
}
