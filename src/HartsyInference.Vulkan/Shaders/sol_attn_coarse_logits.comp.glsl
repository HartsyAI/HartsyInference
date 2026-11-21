// General Sol-Attn in F32, the coarse gate's first stage. SolAttentionReference in Core defines it; sol_attn_f32.cu is
// the CUDA form. Run only when a call gives a gate.
//
// Per (query block, head, batch): the natural-unit score of every key block, the query block's mean query against the
// block's uncentered mean key, [B, H, blocks, blocks]. Ordinary F32 arithmetic, compared by tolerance.

#version 460

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) readonly  buffer C_ { float centroids[]; };
layout(set = 0, binding = 1) readonly  buffer KM_ { float key_means[]; };
layout(set = 0, binding = 2) writeonly buffer L_ { float logits[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint dim;
    uint blocks;
    float scale;
} pc;

void main() {
    uint qb = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint head = b * pc.heads + h;
    uint c = (head * pc.blocks + qb) * pc.dim;
    for (uint n = gl_LocalInvocationID.x; n < pc.blocks; n += 128u) {
        uint k = (head * pc.blocks + n) * pc.dim;
        float dot = 0.0;
        for (uint d = 0u; d < pc.dim; ++d) dot += centroids[c + d] * key_means[k + d];
        logits[(head * pc.blocks + qb) * pc.blocks + n] = dot * pc.scale;
    }
}
