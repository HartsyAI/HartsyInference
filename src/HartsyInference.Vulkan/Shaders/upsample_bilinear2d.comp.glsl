// upsample_bilinear2d: integer-scale bilinear 2D upsample with align_corners=false
// (the diffusers / PyTorch default).
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
#define TO_F32(x) float(x)
#define FROM_F32(x) float16_t(x)
#else
#define DTYPE float
#define TO_F32(x) (x)
#define FROM_F32(x) (x)
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
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total) return;

    uint ow = idx % outW;
    uint t  = idx / outW;
    uint oh = t % outH;
    t = t / outH;
    uint c = t % pc.C;
    uint n = t / pc.C;

    // align_corners = false (PyTorch default for nn.Upsample(mode="bilinear"))
    float src_h = (float(oh) + 0.5) * float(pc.H) / float(outH) - 0.5;
    float src_w = (float(ow) + 0.5) * float(pc.W) / float(outW) - 0.5;

    int h0 = int(floor(src_h));
    int w0 = int(floor(src_w));
    int h1 = h0 + 1;
    int w1 = w0 + 1;
    float fh = src_h - float(h0);
    float fw = src_w - float(w0);

    h0 = clamp(h0, 0, int(pc.H) - 1);
    h1 = clamp(h1, 0, int(pc.H) - 1);
    w0 = clamp(w0, 0, int(pc.W) - 1);
    w1 = clamp(w1, 0, int(pc.W) - 1);

    uint baseNc = (n * pc.C + c) * pc.H;
    float v00 = TO_F32(inp[(baseNc + uint(h0)) * pc.W + uint(w0)]);
    float v01 = TO_F32(inp[(baseNc + uint(h0)) * pc.W + uint(w1)]);
    float v10 = TO_F32(inp[(baseNc + uint(h1)) * pc.W + uint(w0)]);
    float v11 = TO_F32(inp[(baseNc + uint(h1)) * pc.W + uint(w1)]);

    float v0 = v00 * (1.0 - fw) + v01 * fw;
    float v1 = v10 * (1.0 - fw) + v11 * fw;
    float v = v0 * (1.0 - fh) + v1 * fh;
    outp[idx] = FROM_F32(v);
}
