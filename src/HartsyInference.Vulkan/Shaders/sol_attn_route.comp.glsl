// General Sol-Attn in F32, stage 4 of 5 (routing). SolAttentionReference in Core defines it; sol_attn_f32.cu is the
// CUDA form. Operands are token-major [B, T, H, D]; tokens are cut into blocks of 64.
//
// Per (query block, head, batch): which key blocks are read token by token, one uint each, [B, H, blocks, blocks].
// budget < 0 routes by a threshold of tau standard deviations; otherwise by rank: a block is read when fewer than budget
// non-sink blocks score strictly above it, which keeps a group tied at the boundary whole. Neighbours and sink blocks
// are always read, and a sink query block reads everything.
//
// This stage decides the routing, and a routing decision is a comparison. Every value is declared `precise`, so the
// compiler may not contract a multiply and an add into one fused operation, and the operations run in the reference's
// order. Add, subtract and multiply are then the reference's bit for bit, provided the driver keeps subnormal values:
// no float-controls mode is declared here, and without one a driver may flush them to zero. The Vulkan specification
// also does not require a divide or a square root to be correctly rounded, so those are exact only where the driver
// makes them so.
#version 460

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) readonly  buffer P_ { float pooled[]; };
layout(set = 0, binding = 1) readonly  buffer C_ { float centroids[]; };
layout(set = 0, binding = 2) readonly  buffer Var_ { float variance[]; };
layout(set = 0, binding = 3) writeonly buffer R_ { uint reads[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint dim;
    uint blocks;
    float log2s;
    float tau;
    int budget;
    int sinkStart;
    int sinkEnd;
    int sinkQueryStart;
    int sinkQueryEnd;
} pc;

shared float s_threshold;

void main() {
    uint qb = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint head = b * pc.heads + h;
    uint scores = (head * pc.blocks + qb) * pc.blocks;
    if (pc.budget < 0 && gl_LocalInvocationID.x == 0u) {
        uint c = (head * pc.blocks + qb) * pc.dim;
        uint var = head * pc.dim;
        precise float spread = 0.0;
        for (uint d = 0u; d < pc.dim; ++d) {
            precise float square = centroids[c + d] * centroids[c + d];
            precise float term = square * variance[var + d];
            spread = spread + term;
        }
        precise float scaled = spread * pc.log2s;
        precise float scaled2 = scaled * pc.log2s;
        precise float inside = scaled2 + 1e-6;
        precise float limit = pc.tau * sqrt(inside);
        s_threshold = limit;
    }
    barrier();
    bool readsAll = int(qb) >= pc.sinkQueryStart && int(qb) < pc.sinkQueryEnd;
    for (uint n = gl_LocalInvocationID.x; n < pc.blocks; n += 128u) {
        bool sink = int(n) >= pc.sinkStart && int(n) < pc.sinkEnd;
        bool read;
        if (pc.budget < 0) {
            read = pooled[scores + n] > s_threshold;
        } else {
            int above = 0;
            for (uint m = 0u; m < pc.blocks; ++m)
                if (!(int(m) >= pc.sinkStart && int(m) < pc.sinkEnd) && pooled[scores + m] > pooled[scores + n]) ++above;
            read = pc.budget > 0 && !sink && above < pc.budget;
        }
        uint gap = qb > n ? qb - n : n - qb;
        reads[scores + n] = (read || readsAll || gap <= 1u || sink) ? 1u : 0u;
    }
}
