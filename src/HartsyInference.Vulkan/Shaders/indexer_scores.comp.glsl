// indexer_scores: scores[t, n] = scale * sum_h relu(q[t, h] . key[n]) * headWeights[t, h], heads summed in order, matching
// IndexerScoresReference.Apply; -infinity where n >= compressLens[t] or the candidate flag is zero. One thread per score
// element with fusing disabled, so every dot and the head sum round exactly as the reference does. The key row is decoded
// per (score, head, dim), which is the price of keeping the reference's summation order.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Scores_ { float score_data[]; };
layout(set = 0, binding = 1) readonly buffer Query_ { float q_data[]; };
layout(set = 0, binding = 2) readonly buffer Codes_ { uint k_codes[]; };
layout(set = 0, binding = 3) readonly buffer Scales_ { uint k_scales[]; };
layout(set = 0, binding = 4) readonly buffer Weights_ { float w_data[]; };
layout(set = 0, binding = 5) readonly buffer Lens_ { int len_data[]; };
layout(set = 0, binding = 6) readonly buffer Cand_ { uint cand_data[]; };

layout(push_constant) uniform Push {
    uint total;          // tokens * keyRows
    uint keyRows;
    uint heads;
    uint dim;
    uint enc;
    uint hasCandidates;
    float scale;
    uint elemBase;
} pc;

#include "latent_codec.glsl"

LATENT_DECODE_FN(keyAt, k_codes, k_scales)

void main() {
    uint gid = pc.elemBase + gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint t = gid / pc.keyRows;
    uint n = gid - t * pc.keyRows;
    bool masked = int(n) >= len_data[t];
    if (!masked && pc.hasCandidates != 0u) {
        masked = ((cand_data[gid >> 2] >> ((gid & 3u) * 8u)) & 0xFFu) == 0u;
    }
    if (masked) {
        score_data[gid] = uintBitsToFloat(0xFF800000u);
        return;
    }
    precise float sum = 0.0;
    for (uint h = 0u; h < pc.heads; h++) {
        uint qBase = (t * pc.heads + h) * pc.dim;
        precise float dot = 0.0;
        for (uint d = 0u; d < pc.dim; d++) {
            dot += q_data[qBase + d] * keyAt(pc.enc, n, pc.dim, d);
        }
        precise float term = max(dot, 0.0) * w_data[t * pc.heads + h];
        sum += term;
    }
    precise float result = sum * pc.scale;
    score_data[gid] = result;
}
