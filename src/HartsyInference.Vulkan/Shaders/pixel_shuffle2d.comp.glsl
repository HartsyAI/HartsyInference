// pixel_shuffle2d: [N, cOut*r*r, D, H, W] -> [N, cOut, D, H*r, W*r]
//   out[n, c, z, h*r + p1, w*r + p2] = in[n, (c*r + p1)*r + p2, z, h, w]
//
// Driven from the OUTPUT, one invocation per output element, so writes are contiguous and each
// invocation computes the single source it reads. The channel packing is (c*r + p1)*r + p2, which is
// the layout the depth-to-space convention produces; reading it as c + (p1*r + p2)*cOut gives a
// plausible image with the sub-pixel grid scrambled.
//
// Bindings: 0=in, 1=out
#version 460

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer I_ { float inp[];  };
layout(set = 0, binding = 1) writeonly buffer O_ { float outp[]; };

layout(push_constant) uniform Push {
    uint batch;
    uint cOut;
    uint depth;
    uint inH;
    uint inW;
    uint ratio;
    uint total;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;

    uint outW = pc.inW * pc.ratio;
    uint outH = pc.inH * pc.ratio;

    uint ow = i % outW;
    uint r1 = i / outW;
    uint oh = r1 % outH;
    uint r2 = r1 / outH;
    uint z  = r2 % pc.depth;
    uint r3 = r2 / pc.depth;
    uint c  = r3 % pc.cOut;
    uint b  = r3 / pc.cOut;

    uint h = oh / pc.ratio, p1 = oh % pc.ratio;
    uint w = ow / pc.ratio, p2 = ow % pc.ratio;
    uint cin = (c * pc.ratio + p1) * pc.ratio + p2;
    uint cInTotal = pc.cOut * pc.ratio * pc.ratio;

    uint src = (((b * cInTotal + cin) * pc.depth + z) * pc.inH + h) * pc.inW + w;
    outp[i] = inp[src];
}
