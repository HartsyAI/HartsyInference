// transpose: batched 2D transpose [B, D1, D2] -> [B, D2, D1]
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
    uint D1;
    uint D2;
} pc;

void main() {
    uint i  = gl_GlobalInvocationID.x;          // global flat index in output
    uint total = pc.B * pc.D2 * pc.D1;
    if (i >= total) return;
    uint b   = i / (pc.D2 * pc.D1);
    uint rem = i % (pc.D2 * pc.D1);
    uint d2  = rem / pc.D1;
    uint d1  = rem % pc.D1;
    uint inIdx  = b * pc.D1 * pc.D2 + d1 * pc.D2 + d2;
    outp[i] = inp[inIdx];
}
