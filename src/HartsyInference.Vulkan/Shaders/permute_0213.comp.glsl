// permute_0213: [B, S, H, D] -> [B, H, S, D]
// Bindings: 0=in, 1=out
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

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly  buffer In_  { DTYPE inp[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { DTYPE outp[]; };

layout(push_constant) uniform Push {
    uint B;
    uint S;
    uint H;
    uint D;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = pc.B * pc.H * pc.S * pc.D;
    if (idx >= total) return;
    // Output index: (b, h, s, d)
    uint d = idx % pc.D;
    uint t = idx / pc.D;
    uint s = t % pc.S;
    t = t / pc.S;
    uint h = t % pc.H;
    uint b = t / pc.H;
    // Input layout: (b, s, h, d)
    uint inIdx = ((b * pc.S + s) * pc.H + h) * pc.D + d;
    outp[idx] = inp[inIdx];
}
