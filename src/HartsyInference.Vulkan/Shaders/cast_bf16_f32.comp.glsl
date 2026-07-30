// cast_bf16_f32: bfloat16 -> f32. bf16 is the UPPER 16 bits of an IEEE-754 f32 (sign + 8-bit exponent +
// 7-bit mantissa, the lower 16 mantissa bits dropped) — NOT the same format as GLSL's float16_t (IEEE-754
// binary16). Conversion is a zero-extend + left-shift into the f32 bit pattern, no float16_t involved.
//
// Compile:
//   glslc cast_bf16_f32.comp.glsl -o cast_bf16_f32.spv

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly  buffer In_  { uint16_t in_data[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { float out_data[]; };

layout(push_constant) uniform Push {
    uint elements;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.elements) return;
    uint bits = uint(in_data[gid]) << 16;
    out_data[gid] = uintBitsToFloat(bits);
}
