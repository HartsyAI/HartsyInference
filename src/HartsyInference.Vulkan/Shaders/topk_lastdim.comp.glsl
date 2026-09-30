// topk_lastdim: per-row top-k over the last dimension, one workgroup per row, matching TopKReference.
//
// The reference sorts by (NaN last, value descending, index ascending), so the order is total and a pick can be
// found without a taken-set: each round takes the smallest element strictly after the previous pick. Rounds are
// k parallel min-reductions. sortByIndex instead recovers the k-th pick and emits every element at or before it
// in index order with a chunked scan. Slots past min(k, validLength) get index -1 and value -inf.
//
// WGSIZE must equal the dispatch's local_size_x.

#version 460

#define WGSIZE 256

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) writeonly buffer Values_  { float values_data[]; };
layout(set = 0, binding = 1) writeonly buffer Indices_ { int indices_data[]; };
layout(set = 0, binding = 2) readonly buffer Input_    { float input_data[]; };
layout(set = 0, binding = 3) readonly buffer Valid_    { int valid_data[]; };

layout(push_constant) uniform Push {
    uint n;
    uint k;
    uint hasValid;
    uint sortByIndex;
    uint rowBase;
} pc;

shared float sVal[WGSIZE];
shared uint sIdx[WGSIZE];
shared uint sPos[WGSIZE];
shared float sLastVal;
shared uint sLastIdx;
shared uint sRunning;

const uint NONE = 0xFFFFFFFFu;

// True when element (va, ia) sorts strictly before (vb, ib).
bool precedes(float va, uint ia, float vb, uint ib) {
    bool na = isnan(va);
    bool nb = isnan(vb);
    if (na != nb) return nb;
    if (!na && va != vb) return va > vb;
    return ia < ib;
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint row = pc.rowBase + gl_WorkGroupID.x;
    uint n = pc.n;
    uint k = pc.k;
    uint rowBase = row * n;
    uint len = n;
    if (pc.hasValid != 0u) len = uint(clamp(valid_data[row], 0, int(n)));
    uint take = min(k, len);
    uint outBase = row * k;

    if (tid == 0u) { sLastVal = 0.0; sLastIdx = NONE; sRunning = 0u; }
    barrier();

    if (pc.sortByIndex == 0u) {
        for (uint j = 0u; j < take; j++) {
            float lastVal = sLastVal;
            uint lastIdx = sLastIdx;
            float bestVal = 0.0;
            uint bestIdx = NONE;
            for (uint i = tid; i < len; i += WGSIZE) {
                float v = input_data[rowBase + i];
                if (lastIdx != NONE && !precedes(lastVal, lastIdx, v, i)) continue;
                if (bestIdx == NONE || precedes(v, i, bestVal, bestIdx)) { bestVal = v; bestIdx = i; }
            }
            sVal[tid] = bestVal;
            sIdx[tid] = bestIdx;
            barrier();
            for (uint stride = WGSIZE / 2u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    uint oi = sIdx[tid + stride];
                    if (oi != NONE && (sIdx[tid] == NONE || precedes(sVal[tid + stride], oi, sVal[tid], sIdx[tid]))) {
                        sVal[tid] = sVal[tid + stride];
                        sIdx[tid] = oi;
                    }
                }
                barrier();
            }
            if (tid == 0u) {
                indices_data[outBase + j] = int(sIdx[0]);
                values_data[outBase + j] = sVal[0];
                sLastVal = sVal[0];
                sLastIdx = sIdx[0];
            }
            barrier();
        }
    } else if (take > 0u) {
        // k-th pick by the same rounds, then a scan-compaction of everything not after it, in index order.
        for (uint j = 0u; j < take; j++) {
            float lastVal = sLastVal;
            uint lastIdx = sLastIdx;
            float bestVal = 0.0;
            uint bestIdx = NONE;
            for (uint i = tid; i < len; i += WGSIZE) {
                float v = input_data[rowBase + i];
                if (lastIdx != NONE && !precedes(lastVal, lastIdx, v, i)) continue;
                if (bestIdx == NONE || precedes(v, i, bestVal, bestIdx)) { bestVal = v; bestIdx = i; }
            }
            sVal[tid] = bestVal;
            sIdx[tid] = bestIdx;
            barrier();
            for (uint stride = WGSIZE / 2u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    uint oi = sIdx[tid + stride];
                    if (oi != NONE && (sIdx[tid] == NONE || precedes(sVal[tid + stride], oi, sVal[tid], sIdx[tid]))) {
                        sVal[tid] = sVal[tid + stride];
                        sIdx[tid] = oi;
                    }
                }
                barrier();
            }
            if (tid == 0u) { sLastVal = sVal[0]; sLastIdx = sIdx[0]; }
            barrier();
        }
        float kthVal = sLastVal;
        uint kthIdx = sLastIdx;
        for (uint base = 0u; base < len; base += WGSIZE) {
            uint i = base + tid;
            bool picked = false;
            float v = 0.0;
            if (i < len) {
                v = input_data[rowBase + i];
                picked = (i == kthIdx) || precedes(v, i, kthVal, kthIdx);
            }
            sPos[tid] = picked ? 1u : 0u;
            barrier();
            // Hillis-Steele inclusive scan of the flags.
            for (uint off = 1u; off < WGSIZE; off <<= 1u) {
                uint add = (tid >= off) ? sPos[tid - off] : 0u;
                barrier();
                sPos[tid] += add;
                barrier();
            }
            if (picked) {
                uint slot = sRunning + sPos[tid] - 1u;
                indices_data[outBase + slot] = int(i);
                values_data[outBase + slot] = v;
            }
            barrier();
            if (tid == WGSIZE - 1u) sRunning += sPos[tid];
            barrier();
        }
    }

    for (uint j = take + tid; j < k; j += WGSIZE) {
        indices_data[outBase + j] = -1;
        values_data[outBase + j] = uintBitsToFloat(0xFF800000u);
    }
}
