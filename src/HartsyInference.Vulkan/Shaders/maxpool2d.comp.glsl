// maxpool2d: explicit kernel/stride/zero-pad 2D max-pool over NCHW. One invocation per output element.
// out[n,c,oy,ox] = max over (ky,kx) of in[n,c, oy*strideH+ky-padH, ox*strideW+kx-padW]; padded taps skipped.
// Matches IBackend.MaxPool2D (entire-window-OOB → 0). FP32 accumulation even for FP16 I/O.
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
    uint inH;
    uint inW;
    uint outH;
    uint outW;
    uint kernelH;
    uint kernelW;
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
    int iy0 = int(oy * pc.strideH) - int(pc.padH);
    int ix0 = int(ox * pc.strideW) - int(pc.padW);

    float maxVal = -1.0 / 0.0;   // -inf
    bool any = false;
    for (uint ky = 0; ky < pc.kernelH; ky++) {
        int iy = iy0 + int(ky);
        if (iy < 0 || iy >= int(pc.inH)) continue;
        for (uint kx = 0; kx < pc.kernelW; kx++) {
            int ix = ix0 + int(kx);
            if (ix < 0 || ix >= int(pc.inW)) continue;
            float v = float(inp[planeBase + uint(iy) * pc.inW + uint(ix)]);
            if (v > maxVal) maxVal = v;
            any = true;
        }
    }
    outp[i] = DTYPE(any ? maxVal : 0.0);
}
