// depthwise_conv2d: groups==channels 2D conv over NCHW. Weight is [C,1,kH,kW]; bias [C] optional (hasBias).
// out[n,c,oy,ox] = bias[c] + sum_{ky,kx} w[c,0,ky,kx] * in[n,c, oy*strideH+ky-padH, ox*strideW+kx-padW]
// Matches IBackend.Conv2dDepthwise. FP32 accumulation even for FP16 I/O.
// Bindings: 0=in, 1=weight, 2=bias, 3=out
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

layout(set = 0, binding = 0) readonly  buffer In_   { DTYPE inp[]; };
layout(set = 0, binding = 1) readonly  buffer Wgt_  { DTYPE wgt[]; };
layout(set = 0, binding = 2) readonly  buffer Bias_ { DTYPE bias[]; };
layout(set = 0, binding = 3) writeonly buffer Out_  { DTYPE outp[]; };

layout(push_constant) uniform Push {
    uint hasBias;
    uint N;
    uint C;
    uint inH;
    uint inW;
    uint outH;
    uint outW;
    uint kH;
    uint kW;
    uint strideH;
    uint strideW;
    uint padH;
    uint padW;
} pc;

void main() {
    uint total = pc.N * pc.C * pc.outH * pc.outW;
    uint i = gl_GlobalInvocationID.x;
    if (i >= total) return;

    uint ox = i % pc.outW;
    uint t = i / pc.outW;
    uint oy = t % pc.outH;
    t = t / pc.outH;
    uint c = t % pc.C;
    uint n = t / pc.C;

    uint planeBase = (n * pc.C + c) * pc.inH * pc.inW;
    uint kernBase = c * pc.kH * pc.kW;
    int iy0 = int(oy * pc.strideH) - int(pc.padH);
    int ix0 = int(ox * pc.strideW) - int(pc.padW);

    float acc = pc.hasBias != 0u ? float(bias[c]) : 0.0;
    for (uint ky = 0; ky < pc.kH; ky++) {
        int iy = iy0 + int(ky);
        if (iy < 0 || iy >= int(pc.inH)) continue;
        for (uint kx = 0; kx < pc.kW; kx++) {
            int ix = ix0 + int(kx);
            if (ix < 0 || ix >= int(pc.inW)) continue;
            acc += float(wgt[kernBase + ky * pc.kW + kx]) * float(inp[planeBase + uint(iy) * pc.inW + uint(ix)]);
        }
    }
    outp[i] = DTYPE(acc);
}
