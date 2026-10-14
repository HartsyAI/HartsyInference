// sdpa_flash_decode_combine: flash-decoding pass 2 for ONE query row. One workgroup per (query head, batch) merges
// the partial softmax states that sdpa_flash_decode_split wrote for the LIVE splits only (the same live count the
// split pass derives from the same range and chunk): M = max m_s, L = sum l_s * exp(m_s - M), and
// out = sum acc_s * exp(m_s - M) / L. A row with no valid key writes zeros, which is what the scalar kernel writes.
//
// Same geometry and lane layout as the split pass; LANES (spec constant 0) = subgroup size, DPL (spec constant 2) =
// headDim / LANES. The combine itself needs no subgroup operation.

#version 460

#ifndef HAS_DEVICE_POS
#define HAS_DEVICE_POS 0
#endif

layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint LANES = 32;
layout(constant_id = 2) const uint DPL = 2;

#define MAX_DPL 4
#define NEG_INF (-1.0 / 0.0)

layout(set = 0, binding = 0) readonly buffer Part_ { float part_data[]; };
#if HAS_DEVICE_POS == 1
layout(set = 0, binding = 1) readonly  buffer Pos_ { uint pos_data[]; };   // [0] = kvLen, [1] = qOffset
layout(set = 0, binding = 2) writeonly buffer O_ { float o_data[]; };
#else
layout(set = 0, binding = 1) writeonly buffer O_ { float o_data[]; };
#endif

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint kvCapacity;     // unused here; kept so both passes share one push layout
    uint skv;            // eager only
    uint qOffset;        // eager only
    uint headDim;
    float scale;         // unused here
    uint causal;
    uint slidingWindow;
    uint splits;
    uint chunk;
} pc;

void main() {
    uint h = gl_WorkGroupID.x;
    uint b = gl_WorkGroupID.y;
    uint lane = gl_LocalInvocationID.x;
    uint D = pc.headDim;

#if HAS_DEVICE_POS == 1
    uint skv = pos_data[0];
    uint qPos = pos_data[1];
#else
    uint skv = pc.skv;
    uint qPos = pc.qOffset;
#endif

    uint hi = skv;
    if (pc.causal != 0u || pc.slidingWindow != 0u) hi = min(skv, qPos + 1u);
    uint lo = 0u;
    if (pc.slidingWindow != 0u && qPos + 1u > pc.slidingWindow) lo = qPos + 1u - pc.slidingWindow;
    uint rangeLen = hi > lo ? hi - lo : 0u;
    uint live = (rangeLen + pc.chunk - 1u) / pc.chunk;

    uint S = pc.splits;
    uint rowBase = (b * pc.hq + h) * S;

    float gMax = NEG_INF;
    for (uint s = 0u; s < live; s++) gMax = max(gMax, part_data[(rowBase + s) * (D + 2u)]);

    float denom = 0.0;
    float acc[MAX_DPL];
    for (uint j = 0u; j < MAX_DPL; j++) acc[j] = 0.0;
    if (gMax > NEG_INF) {
        for (uint s = 0u; s < live; s++) {
            uint base = (rowBase + s) * (D + 2u);
            float w = exp(part_data[base] - gMax);
            denom += part_data[base + 1u] * w;
            for (uint j = 0u; j < DPL; j++) acc[j] += w * part_data[base + 2u + j * LANES + lane];
        }
    }

    float inv = denom > 0.0 ? 1.0 / denom : 0.0;
    uint outBase = (b * pc.hq + h) * D;
    for (uint j = 0u; j < DPL; j++) o_data[outBase + j * LANES + lane] = acc[j] * inv;
}
