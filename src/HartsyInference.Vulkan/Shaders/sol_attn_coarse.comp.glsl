// General Sol-Attn in F32, the coarse gate's second stage. Run only when a call gives a gate.
//
// Per (query block, head, batch): one softmax over key blocks of that query block's coarse logits, mixing the blocks'
// mean values (the summed V over the live length), [B, H, blocks, D]. The token pass adds gate * coarse to each row.

#version 460

layout(local_size_x = 128) in;

#define NEG_INF (-1.0 / 0.0)

layout(set = 0, binding = 0) readonly  buffer L_ { float logits[]; };
layout(set = 0, binding = 1) readonly  buffer VS_ { float value_sums[]; };
layout(set = 0, binding = 2) readonly  buffer Len_ { int len_data[]; };
layout(set = 0, binding = 3) writeonly buffer Co_ { float coarse[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint dim;
    uint blocks;
} pc;

void main() {
    uint qb = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint head = b * pc.heads + h;
    uint row = (head * pc.blocks + qb) * pc.blocks;
    float m = NEG_INF;
    for (uint n = 0u; n < pc.blocks; ++n) m = max(m, logits[row + n]);
    float total = 0.0;
    for (uint n = 0u; n < pc.blocks; ++n) total += exp(logits[row + n] - m);
    for (uint d = gl_LocalInvocationID.x; d < pc.dim; d += 128u) {
        float sum = 0.0;
        for (uint n = 0u; n < pc.blocks; ++n)
            sum += exp(logits[row + n] - m) / total / float(len_data[n]) * value_sums[(head * pc.blocks + n) * pc.dim + d];
        coarse[(head * pc.blocks + qb) * pc.dim + d] = sum;
    }
}
