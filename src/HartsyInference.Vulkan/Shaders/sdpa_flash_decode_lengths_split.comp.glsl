// sdpa_flash_decode_lengths_split: flash-decoding pass 1 for a decode whose sequences have their own key lengths
// (AttentionCall.SequenceLengths, docs/ATTENTION_CONTRACT.md section 2). sdpa_flash_decode_split with three changes:
// the valid length is lengths[b] clamped to the cache capacity, a workgroup serves query row r of sequence b
// (gl_WorkGroupID.z = b * S + r), and there is no window and no device position.
//
// Causal: the step's S rows are the sequence's last S slots, so row r reads the slots below lengths[b] - S + r + 1, and
// none when that is not positive. Not causal: every row reads all lengths[b] slots.
//
// The partial layout is that shader's with the row index (b * S + r) * hq + h, which is the row's place in the
// [B, S, hq * D] output. A split past the row's live count does no work and writes nothing;
// sdpa_flash_decode_lengths_combine derives the same live count and reads only the live splits.
//
// The cache is F32, F16 or BF16 (keys and values of one type, CACHE below); Q, the partial states and the output are
// F32. A half-type element is widened as it is read, exactly, so the arithmetic is the F32 kernel's over the stored
// values, as in the CUDA kernel.
//
// The workgroup is exactly one subgroup: LANES (spec constant 0) = the device subgroup size.

#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

// LANES = workgroup width = subgroup size. GROUP = query heads per KV head. DPL = head dims per lane = headDim / LANES.
layout(local_size_x_id = 0) in;
layout(constant_id = 0) const uint LANES = 32;
layout(constant_id = 1) const uint GROUP = 1;
layout(constant_id = 2) const uint DPL = 2;
layout(constant_id = 3) const uint CACHE = 0;   // 0: F32 keys and values; 1: F16; 2: BF16

// The register arrays below are sized by GROUP and DPL themselves, so a specialization holds only what it uses: a head
// dim of two lane widths keeps two dims a lane however large the widest admitted call is. The host admits a group of
// at most 8 and at most 8 dims a lane (VulkanBackend.FlashDecodeMaxGroup, FlashDecodeLengthsMaxDpl).
#define NEG_INF (-1.0 / 0.0)

layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
// Keys and values as words: one F32 element a word, or two half-type elements a word, the lower 16 bits first.
layout(set = 0, binding = 1) readonly  buffer K_ { uint k_bits[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { uint v_bits[]; };
layout(set = 0, binding = 3) readonly  buffer Len_ { int len_data[]; };     // one valid key length per sequence
layout(set = 0, binding = 4) writeonly buffer Part_ { float part_data[]; };

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint kvCapacity;     // per-head stride of the K/V buffers (keys)
    uint rows;           // S: query rows per sequence
    uint headDim;
    float scale;
    uint causal;         // 0/1
    uint splits;         // split count of the grid (partial-state stride)
    uint chunk;          // keys per split
} pc;

// The word holding cache element e, and that element widened to F32. F16 through unpackHalf2x16 and BF16 by placing
// its bits in the upper half of an F32: both exact.
uint cacheWord(uint e) { return CACHE == 0u ? e : e >> 1u; }
float widen(uint word, uint e) {
    if (CACHE == 0u) return uintBitsToFloat(word);
    uint h = (word >> (16u * (e & 1u))) & 0xFFFFu;
    return CACHE == 1u ? unpackHalf2x16(h).x : uintBitsToFloat(h << 16u);
}

void main() {
    uint split = gl_WorkGroupID.x;
    uint kvHead = gl_WorkGroupID.y;
    uint z = gl_WorkGroupID.z;
    uint b = z / pc.rows;
    uint r = z % pc.rows;
    uint lane = gl_LocalInvocationID.x;
    uint D = pc.headDim;

    // Valid keys [0, hi): the sequence's own length, and under the causal rule only the slots up to this row's position.
    int len = clamp(len_data[b], 0, int(pc.kvCapacity));
    int bound = pc.causal != 0u ? len - int(pc.rows) + int(r) + 1 : len;
    uint hi = bound > 0 ? uint(bound) : 0u;
    uint live = (hi + pc.chunk - 1u) / pc.chunk;
    if (split >= live) return;   // dead split: no work, no partial state

    uint keyBeg = split * pc.chunk;
    uint keyEnd = min((split + 1u) * pc.chunk, hi);

    float qv[GROUP][DPL];
    float acc[GROUP][DPL];
    float runMax[GROUP];
    float runSum[GROUP];
    for (uint g = 0u; g < GROUP; g++) {
        uint qRow = ((b * pc.hq + kvHead * GROUP + g) * pc.rows + r) * D;
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
        float kr[4][DPL];
        float vr[4][DPL];
        for (uint u = 0u; u < 4u; u++) {
            for (uint j = 0u; j < DPL; j++) {
                kr[u][j] = 0.0;
                vr[u][j] = 0.0;
            }
            if (u < cnt) {
                uint row = (kvRow0 + k + u) * D;
                for (uint j = 0u; j < DPL; j++) {
                    uint e = row + j * LANES + lane;
                    kr[u][j] = widen(k_bits[cacheWord(e)], e);
                    vr[u][j] = widen(v_bits[cacheWord(e)], e);
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

    // Partial layout per (sequence and row, query head, split): [m, l, acc[0..D)], stride D + 2.
    uint S = pc.splits;
    for (uint g = 0u; g < GROUP; g++) {
        uint h = kvHead * GROUP + g;
        uint base = ((z * pc.hq + h) * S + split) * (D + 2u);
        for (uint j = 0u; j < DPL; j++) part_data[base + 2u + j * LANES + lane] = acc[g][j];
        if (lane == 0u) {
            part_data[base] = runMax[g];
            part_data[base + 1u] = runSum[g];
        }
    }
}
