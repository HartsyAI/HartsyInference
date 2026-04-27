// upsample_nearest2d: integer-scale nearest-neighbor 2D upsample.
// in : [N, C, H, W]   out : [N, C, H*scaleH, W*scaleW]
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

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer In_  { DTYPE inp[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { DTYPE outp[]; };

layout(push_constant) uniform Push {
    uint N;
    uint C;
    uint H;
    uint W;
    uint scaleH;
    uint scaleW;
} pc;

void main() {
    uint outH = pc.H * pc.scaleH;
    uint outW = pc.W * pc.scaleW;
    uint total = pc.N * pc.C * outH * outW;
    uint i = gl_GlobalInvocationID.x;
    if (i >= total) return;

    uint w = i % outW;
    uint t = i / outW;
    uint h = t % outH;
    t = t / outH;
    uint c = t % pc.C;
    uint n = t / pc.C;

    uint inH = h / pc.scaleH;
    uint inW = w / pc.scaleW;
    uint inIdx = ((n * pc.C + c) * pc.H + inH) * pc.W + inW;
    outp[i] = inp[inIdx];
}
