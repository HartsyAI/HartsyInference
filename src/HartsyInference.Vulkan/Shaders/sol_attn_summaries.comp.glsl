// General Sol-Attn in F32, stage 1 of 5 (block summaries). SolAttentionReference in Core defines it; sol_attn_f32.cu is the
// CUDA form. Operands are token-major [B, T, H, D]; tokens are cut into blocks of 64.
//
// Per (block, head, batch): mean K, summed V and mean Q over the block's live rows, each [B, H, blocks, D].
//
// This stage decides the routing, and a routing decision is a comparison. Every value is declared `precise`, so the
// compiler may not contract a multiply and an add into one fused operation, and the operations run in the reference's
// order. Add, subtract and multiply are then the reference's bit for bit, provided the driver keeps subnormal values:
// no float-controls mode is declared here, and without one a driver may flush them to zero. The Vulkan specification
// also does not require a divide or a square root to be correctly rounded, so those are exact only where the driver
// makes them so.
#version 460

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { float k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { float v_data[]; };
layout(set = 0, binding = 3) readonly  buffer Len_ { int len_data[]; };
layout(set = 0, binding = 4) writeonly buffer KM_ { float key_means[]; };
layout(set = 0, binding = 5) writeonly buffer VS_ { float value_sums[]; };
layout(set = 0, binding = 6) writeonly buffer C_ { float centroids[]; };

layout(push_constant) uniform Push {
    uint tokens;
    uint heads;
    uint dim;
    uint blocks;
} pc;

uint row(uint b, uint t, uint h) { return ((b * pc.tokens + t) * pc.heads + h) * pc.dim; }

void main() {
    uint n = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint len = uint(len_data[n]);
    uint outBase = ((b * pc.heads + h) * pc.blocks + n) * pc.dim;
    for (uint d = gl_LocalInvocationID.x; d < pc.dim; d += 128u) {
        precise float ks = 0.0;
        precise float vs = 0.0;
        precise float qs = 0.0;
        for (uint t = n * 64u; t < n * 64u + len; ++t) {
            uint at = row(b, t, h) + d;
            ks = ks + k_data[at];
            vs = vs + v_data[at];
            qs = qs + q_data[at];
        }
        precise float km = ks / float(len);
        precise float c = qs / float(len);
        key_means[outBase + d] = km;
        value_sums[outBase + d] = vs;
        centroids[outBase + d] = c;
    }
}
