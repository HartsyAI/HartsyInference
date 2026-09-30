// act_quant_dequant: replaces each scale group of an F32 tensor with its quantize-then-dequantize round trip in place,
// matching LatentQuantReference.ActQuantDequantInPlace bit for bit. One thread per group of 32 (fp8, fp4 e8m0) or 16 (fp4 e4m3).

#version 460
#extension GL_GOOGLE_include_directive : require

#define LATENT_ENCODE

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer X_ { float x_data[]; };

layout(push_constant) uniform Push {
    uint groups;
    uint enc;
    uint groupBase;
} pc;

#include "div_rn.glsl"
#include "latent_codec.glsl"

void main() {
    uint g = pc.groupBase + gl_GlobalInvocationID.x;
    if (g >= pc.groups) return;
    uint group = latentGroupSize(pc.enc);
    uint base = g * group;
    float amax = 0.0;
    for (uint e = 0u; e < group; e++) amax = max(amax, abs(x_data[base + e]));
    float scale;
    latentScaleByte(pc.enc, amax, scale);
    for (uint e = 0u; e < group; e++) {
        uint code = latentCode(pc.enc, x_data[base + e], scale);
        precise float restored = latentDecodeCode(pc.enc, code) * scale;
        x_data[base + e] = restored;
    }
}
