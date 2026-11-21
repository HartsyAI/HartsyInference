// sdpa_decode_merge: the speculative-decode merge (DecodeMergeReference in Core; flash_attn_decode_merge.cu on CUDA).
// Folds a verify step's own rows into an attention result computed over the committed prefix alone.
//
// One workgroup per output row: gl_WorkGroupID = (head h, step row r, sequence b). The prefix result carries the weight
// exp(lse - m), step row t the weight exp(score_t - m) when bit t of mask[r] is set, with
// score_t = scale * q[b, h, r, :] . k[b, hk, t, :] and m = max(lse, largest named score). The merged row is the weighted sum
// of the prefix result and the named step values over the sum of the weights. A row with no key at all (lse == -inf and no
// bit set) is zeros. A prefix with lse == -inf is not read, and neither is a step row no bit names.
//
// Layouts: Q [B, Hq, S, D]; step K and V [B, Hkv, S, D]; prefix and merged [B, S, Hq * D] (two buffers, never the same
// one); lse [B, Hq, S]; mask [S]. All F32 but the mask. A step holds at most 32 rows: invocation t scores step row t.
//
// No subgroup operation: the 32 scores go through shared memory, so the kernel does not depend on the subgroup size.

#version 460

layout(local_size_x = 32) in;

#define NEG_INF (-1.0 / 0.0)

layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { float k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { float v_data[]; };
layout(set = 0, binding = 3) readonly  buffer P_ { float p_data[]; };
layout(set = 0, binding = 4) readonly  buffer L_ { float l_data[]; };
layout(set = 0, binding = 5) readonly  buffer M_ { int m_data[]; };
layout(set = 0, binding = 6) writeonly buffer O_ { float o_data[]; };

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint rows;      // S
    uint headDim;   // D
    float scale;
} pc;

shared float s_score[32];
shared float s_weight[32];

void main() {
    uint h = gl_WorkGroupID.x;
    uint r = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint lane = gl_LocalInvocationID.x;
    uint S = pc.rows;
    uint D = pc.headDim;
    uint hk = h / (pc.hq / pc.hkv);
    uint qBase = ((b * pc.hq + h) * S + r) * D;
    uint kvBase = (b * pc.hkv + hk) * S * D;
    uint outBase = ((b * S + r) * pc.hq + h) * D;
    float rowLse = l_data[(b * pc.hq + h) * S + r];
    int rowMask = m_data[r];
    bool named = lane < S && ((rowMask >> int(lane)) & 1) != 0;

    float s = NEG_INF;
    if (named) {
        uint kAt = kvBase + lane * D;
        float dot = 0.0;
        for (uint d = 0u; d < D; ++d) dot += q_data[qBase + d] * k_data[kAt + d];
        s = dot * pc.scale;
    }
    s_score[lane] = s;
    barrier();

    float m = rowLse;
    for (uint t = 0u; t < 32u; ++t) m = max(m, s_score[t]);
    bool any = m > NEG_INF;
    s_weight[lane] = named && any ? exp(s - m) : 0.0;
    barrier();

    float wp = any && rowLse > NEG_INF ? exp(rowLse - m) : 0.0;
    float total = wp;
    for (uint t = 0u; t < 32u; ++t) total += s_weight[t];

    for (uint d = lane; d < D; d += 32u) {
        float a = wp > 0.0 ? p_data[outBase + d] * wp : 0.0;
        for (uint t = 0u; t < S; ++t)
            if (((rowMask >> int(t)) & 1) != 0) a += s_weight[t] * v_data[kvBase + t * D + d];
        o_data[outBase + d] = total > 0.0 ? a / total : 0.0;
    }
}
