// cast_f32_bf16: f32 -> bfloat16, round-to-nearest-even (the standard bf16 truncation convention: add a
// rounding bias before dropping the lower 16 mantissa bits, so exact halfway cases round to even rather
// than always down). See cast_bf16_f32.comp.glsl for the reverse direction and format note.
//
// Compile:
//   glslc cast_f32_bf16.comp.glsl -o cast_f32_bf16.spv

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly  buffer In_  { float in_data[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { uint16_t out_data[]; };

layout(push_constant) uniform Push {
    uint elements;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.elements) return;
    uint bits = floatBitsToUint(in_data[gid]);
    // NaN must stay NaN after truncation (round-to-nearest-even could otherwise carry a NaN's mantissa
    // into the exponent field on overflow) — force the canonical quiet-NaN upper half instead.
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        out_data[gid] = uint16_t(0x7FC0u);
        return;
    }
    uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    out_data[gid] = uint16_t(rounded >> 16);
}
