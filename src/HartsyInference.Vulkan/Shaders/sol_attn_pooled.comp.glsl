// General Sol-Attn in F32, stage 3 of 5 (pooled scores). SolAttentionReference in Core defines it; sol_attn_f32.cu is the
// CUDA form. Operands are token-major [B, T, H, D]; tokens are cut into blocks of 64.
//
// Per (query block, head, batch): each key block's pooled score, the query block's mean query against the centered
// key mean, in base-2 units, [B, H, blocks, blocks].
//
// This stage decides the routing, and a routing decision is a comparison. Every value is declared `precise`, so the
// compiler may not contract a multiply and an add into one fused operation, and the operations run in the reference's
// order. Add, subtract and multiply are then the reference's bit for bit, provided the driver keeps subnormal values:
// no float-controls mode is declared here, and without one a driver may flush them to zero. The Vulkan specification
// also does not require a divide or a square root to be correctly rounded, so those are exact only where the driver
// makes them so.
#version 460

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) readonly  buffer C_ { float centroids[]; };
layout(set = 0, binding = 1) readonly  buffer KM_ { float key_means[]; };
layout(set = 0, binding = 2) readonly  buffer M_ { float key_mean[]; };
layout(set = 0, binding = 3) writeonly buffer P_ { float pooled[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint dim;
    uint blocks;
    float log2s;
} pc;

void main() {
    uint qb = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint head = b * pc.heads + h;
    uint c = (head * pc.blocks + qb) * pc.dim;
    uint mean = head * pc.dim;
    for (uint n = gl_LocalInvocationID.x; n < pc.blocks; n += 128u) {
        uint k = (head * pc.blocks + n) * pc.dim;
        precise float dot = 0.0;
        for (uint d = 0u; d < pc.dim; ++d) {
            precise float centered = key_means[k + d] - key_mean[mean + d];
            precise float product = centroids[c + d] * centered;
            dot = dot + product;
        }
        precise float score = dot * pc.log2s;
        pooled[(head * pc.blocks + qb) * pc.blocks + n] = score;
    }
}
