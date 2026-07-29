// cast_f16_f32: float16_t -> float elementwise cast.
// Bindings: 0=in (f16), 1=out (f32)
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer In_  { float16_t inp[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { float     outp[]; };

layout(push_constant) uniform Push { uint count; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    outp[i] = float(inp[i]);
}
