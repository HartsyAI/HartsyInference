// moe_act_mul_f16: out[i] = f16(act(gate[i]) * up[i]) with F32 gate and up (the per-expert GEMM outputs), SiLU or tanh-GELU.
// Bindings: 0=out (f16), 1=gate (f32), 2=up (f32).
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) writeonly buffer Out_  { float16_t out_data[]; };
layout(set = 0, binding = 1) readonly  buffer Gate_ { float gate_data[]; };
layout(set = 0, binding = 2) readonly  buffer Up_   { float up_data[]; };

layout(push_constant) uniform Push { uint count; uint gelu; } pc;

float siluf(float x) { return x / (1.0 + exp(-x)); }
float geluTanh(float x) { return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    float g = gate_data[i];
    out_data[i] = float16_t((pc.gelu != 0u ? geluTanh(g) : siluf(g)) * up_data[i]);
}
