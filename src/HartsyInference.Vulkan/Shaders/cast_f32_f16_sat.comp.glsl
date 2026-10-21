// cast_f32_f16_sat: float -> float16_t, saturating to +-65504 as cuDNN's attention casts do; a NaN stays NaN.
// Attention operands only: every other cast keeps cast_f32_f16.
// Bindings: 0=in (f32), 1=out (f16)
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer In_  { float  inp[];  };
layout(set = 0, binding = 1) writeonly buffer Out_ { float16_t outp[]; };

layout(push_constant) uniform Push { uint count; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    float x = inp[i];
    bool nan = (floatBitsToUint(x) & 0x7fffffffu) > 0x7f800000u;   // on the bits: a folded min/max would drop the NaN
    outp[i] = float16_t(nan ? x : clamp(x, -65504.0, 65504.0));
}
