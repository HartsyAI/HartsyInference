// sdpa_flash_decode_split: flash-decoding pass 1 for ONE query row (the decode step). The valid key range of the
// row is cut into fixed-length splits; one workgroup takes one (split, KV head, batch) and produces the partial
// softmax state (running max m, running sum l, unnormalized weighted value acc) for EVERY query head of that KV
// group, so each K/V row is read once per KV head rather than once per query head. sdpa_flash_decode_combine
// merges the live splits.
//
// Geometry is a pure function of the cache capacity (chunk, split count), so eager and graph decode run the same
// arithmetic. A split past the live length (ceil(rangeLen / chunk)) does no work and writes nothing.
//
// The workgroup is exactly one subgroup: LANES (spec constant 0) = the device subgroup size, so subgroupAdd reduces
// one key's dot product across the whole group. Lane l owns head dims l + LANES*j, so every K/V load a warp issues
// is one contiguous run of LANES floats. Masks are the scalar kernel's: causal (keys above the query position) and a
// sliding window (keys at or below qPos - window); the range below encodes both, so no per-key test is needed.

#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#ifndef HAS_DEVICE_POS
#define HAS_DEVICE_POS 0
#endif

// LANES = workgroup width = subgroup size. GROUP = query heads per KV head. DPL = head dims per lane = headDim / LANES.
layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint LANES = 32;
layout(constant_id = 1) const uint GROUP = 1;
layout(constant_id = 2) const uint DPL = 2;

#define MAX_GROUP 8
#define MAX_DPL 4
#define NEG_INF (-1.0 / 0.0)

layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { float k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { float v_data[]; };
#if HAS_DEVICE_POS == 1
layout(set = 0, binding = 3) readonly  buffer Pos_ { uint pos_data[]; };   // [0] = kvLen, [1] = qOffset
layout(set = 0, binding = 4) writeonly buffer Part_ { float part_data[]; };
#else
layout(set = 0, binding = 3) writeonly buffer Part_ { float part_data[]; };
#endif

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint kvCapacity;     // per-head stride of the K/V buffers (keys)
    uint skv;            // eager only: valid keys
    uint qOffset;        // eager only: absolute position of the query row
    uint headDim;
    float scale;
    uint causal;         // 0/1
    uint slidingWindow;  // 0 = disabled
    uint splits;         // split count of the grid (partial-state stride)
    uint chunk;          // keys per split
} pc;

void main() {
    uint split = gl_WorkGroupID.x;
    uint kvHead = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint lane = gl_LocalInvocationID.x;
    uint D = pc.headDim;

#if HAS_DEVICE_POS == 1
    uint skv = min(pos_data[0], pc.kvCapacity);
    uint qPos = pos_data[1];
#else
    uint skv = pc.skv;
    uint qPos = pc.qOffset;
#endif

    // Valid keys [lo, hi): causal drops keys above qPos; a window drops keys at or below qPos - window.
    uint hi = skv;
    if (pc.causal != 0u) hi = min(skv, qPos + 1u);
    uint lo = 0u;
    if (pc.slidingWindow != 0u && qPos + 1u > pc.slidingWindow) lo = qPos + 1u - pc.slidingWindow;
    uint rangeLen = hi > lo ? hi - lo : 0u;
    uint live = (rangeLen + pc.chunk - 1u) / pc.chunk;
    if (split >= live) return;   // dead split: no work, no partial state

    uint keyBeg = lo + split * pc.chunk;
    uint keyEnd = min(lo + (split + 1u) * pc.chunk, hi);

    float qv[MAX_GROUP][MAX_DPL];
    float acc[MAX_GROUP][MAX_DPL];
    float runMax[MAX_GROUP];
    float runSum[MAX_GROUP];
    for (uint g = 0u; g < GROUP; g++) {
        uint qRow = (b * pc.hq + kvHead * GROUP + g) * D;
        for (uint j = 0u; j < DPL; j++) {
            qv[g][j] = q_data[qRow + j * LANES + lane];
            acc[g][j] = 0.0;
        }
        runMax[g] = NEG_INF;
        runSum[g] = 0.0;
    }

    uint kvRow0 = (b * pc.hkv + kvHead) * pc.kvCapacity;
    // Four keys per step: their K/V loads are issued together, then folded into the running state in key order.
    for (uint k = keyBeg; k < keyEnd; k += 4u) {
        uint cnt = min(4u, keyEnd - k);
        float kr[4][MAX_DPL];
        float vr[4][MAX_DPL];
        for (uint u = 0u; u < 4u; u++) {
            for (uint j = 0u; j < DPL; j++) {
                kr[u][j] = 0.0;
                vr[u][j] = 0.0;
            }
            if (u < cnt) {
                uint row = (kvRow0 + k + u) * D;
                for (uint j = 0u; j < DPL; j++) {
                    kr[u][j] = k_data[row + j * LANES + lane];
                    vr[u][j] = v_data[row + j * LANES + lane];
                }
            }
        }
        for (uint u = 0u; u < 4u; u++) {
            if (u >= cnt) break;
            for (uint g = 0u; g < GROUP; g++) {
                float part = 0.0;
                for (uint j = 0u; j < DPL; j++) part += qv[g][j] * kr[u][j];
                float score = subgroupAdd(part) * pc.scale;
                float mNew = max(runMax[g], score);
                float corr = exp(runMax[g] - mNew);
                float p = exp(score - mNew);
                runSum[g] = runSum[g] * corr + p;
                for (uint j = 0u; j < DPL; j++) acc[g][j] = acc[g][j] * corr + p * vr[u][j];
                runMax[g] = mNew;
            }
        }
    }

    // Partial layout per (batch, query head, split): [m, l, acc[0..D)], stride D + 2.
    uint S = pc.splits;
    for (uint g = 0u; g < GROUP; g++) {
        uint h = kvHead * GROUP + g;
        uint base = ((b * pc.hq + h) * S + split) * (D + 2u);
        for (uint j = 0u; j < DPL; j++) part_data[base + 2u + j * LANES + lane] = acc[g][j];
        if (lane == 0u) {
            part_data[base] = runMax[g];
            part_data[base + 1u] = runSum[g];
        }
    }
}
