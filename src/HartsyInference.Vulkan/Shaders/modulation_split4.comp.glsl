// modulation_split4: AdaLN modulation split.
//   proj [B, 4D] -> scaleMsa/gateMsa/scaleMlp/gateMlp, each [B, D]
//   scales get 1 + x, gates get tanh(x).
//
// The 1+x on scales and tanh on gates are not interchangeable and not decoration: a scale is a residual
// around identity, so zero must mean "leave it alone"; a gate is bounded to (-1, 1) so a block can be
// turned off smoothly. Swapping them produces plausible-looking output that is wrong everywhere.
//
// Bindings: 0=proj (in), 1=scaleMsa, 2=gateMsa, 3=scaleMlp, 4=gateMlp (out)
#version 460

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer P_  { float proj[];     };
layout(set = 0, binding = 1) writeonly buffer SA_ { float scaleMsa[]; };
layout(set = 0, binding = 2) writeonly buffer GA_ { float gateMsa[];  };
layout(set = 0, binding = 3) writeonly buffer SL_ { float scaleMlp[]; };
layout(set = 0, binding = 4) writeonly buffer GL_ { float gateMlp[];  };

layout(push_constant) uniform Push {
    uint dim;
    uint batch;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = pc.batch * pc.dim;
    if (gid >= total) return;

    uint b = gid / pc.dim;
    uint d = gid - b * pc.dim;
    uint src = b * 4u * pc.dim;

    scaleMsa[gid] = 1.0 + proj[src + d];
    gateMsa[gid]  = tanh(proj[src + pc.dim + d]);
    scaleMlp[gid] = 1.0 + proj[src + 2u * pc.dim + d];
    gateMlp[gid]  = tanh(proj[src + 3u * pc.dim + d]);
}
