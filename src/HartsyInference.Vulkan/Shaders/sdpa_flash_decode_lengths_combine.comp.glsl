// sdpa_flash_decode_lengths_combine: flash-decoding pass 2 for a decode whose sequences have their own key lengths.
// One workgroup per (query head, sequence and row) merges the partial softmax states sdpa_flash_decode_lengths_split
// wrote for the LIVE splits only (the same live count, from the same length and chunk): M = max m_s,
// L = sum l_s * exp(m_s - M), out = sum acc_s * exp(m_s - M) / L, and LSE = M + log(L). The output is written as
// [B, S, hq * D] and the LSE as [B, hq, S]. A row with no valid key writes zeros and an LSE of -inf.
//
// Same geometry and lane layout as the split pass; the combine itself needs no subgroup operation.

#version 460

layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint LANES = 32;
layout(constant_id = 2) const uint DPL = 2;

// The accumulator is sized by DPL itself; the host admits at most 8 dims a lane.
#define NEG_INF (-1.0 / 0.0)

layout(set = 0, binding = 0) readonly buffer Part_ { float part_data[]; };
layout(set = 0, binding = 1) readonly  buffer Len_ { int len_data[]; };
layout(set = 0, binding = 2) writeonly buffer O_ { float o_data[]; };
layout(set = 0, binding = 3) writeonly buffer Lse_ { float lse_data[]; };   // always bound; a scratch buffer when not asked for

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;            // unused here
    uint kvCapacity;     // clamps a length; same layout as the split pass
    uint rows;           // S
    uint headDim;
    float scale;         // unused here
    uint causal;
    uint splits;
    uint chunk;
} pc;

void main() {
    uint h = gl_WorkGroupID.x;
    uint z = gl_WorkGroupID.y;
    uint b = z / pc.rows;
    uint r = z % pc.rows;
    uint lane = gl_LocalInvocationID.x;
    uint D = pc.headDim;

    int len = clamp(len_data[b], 0, int(pc.kvCapacity));
    int bound = pc.causal != 0u ? len - int(pc.rows) + int(r) + 1 : len;
    uint hi = bound > 0 ? uint(bound) : 0u;
    uint live = (hi + pc.chunk - 1u) / pc.chunk;

    uint S = pc.splits;
    uint rowBase = (z * pc.hq + h) * S;

    float gMax = NEG_INF;
    for (uint s = 0u; s < live; s++) gMax = max(gMax, part_data[(rowBase + s) * (D + 2u)]);

    float denom = 0.0;
    float acc[DPL];
    for (uint j = 0u; j < DPL; j++) acc[j] = 0.0;
    if (gMax > NEG_INF) {
        for (uint s = 0u; s < live; s++) {
            uint base = (rowBase + s) * (D + 2u);
            float w = exp(part_data[base] - gMax);
            denom += part_data[base + 1u] * w;
            for (uint j = 0u; j < DPL; j++) acc[j] += w * part_data[base + 2u + j * LANES + lane];
        }
    }

    float inv = denom > 0.0 ? 1.0 / denom : 0.0;
    uint outBase = (z * pc.hq + h) * D;
    for (uint j = 0u; j < DPL; j++) o_data[outBase + j * LANES + lane] = acc[j] * inv;
    if (lane == 0u) lse_data[(b * pc.hq + h) * pc.rows + r] = denom > 0.0 ? gMax + log(denom) : NEG_INF;
}
