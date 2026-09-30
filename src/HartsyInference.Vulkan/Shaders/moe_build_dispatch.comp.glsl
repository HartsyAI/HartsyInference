// moe_build_dispatch: histogram, exclusive scan and stable expert-major permutation of the (token, slot) pairs,
// matching MoeReference.BuildDispatch. ONE workgroup, because the permutation must be identical on every run and
// every device: pairs keep their flat order inside an expert. That is done without atomics on the placement: each
// chunk of WGSIZE pairs ranks its members against the earlier members of the same chunk, and the last one per
// expert advances that expert's running count. An index outside [0, E) drops its pair (pairSlot = -1).
//
// WGSIZE must equal the dispatch's local_size_x.

#version 460

#define WGSIZE 256

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) coherent buffer Counts_   { uint counts[]; };
layout(set = 0, binding = 1) coherent buffer Offsets_  { uint offsets[]; };
layout(set = 0, binding = 2) writeonly buffer Permuted_ { int permuted_token[]; };
layout(set = 0, binding = 3) writeonly buffer PairSlot_ { int pair_slot[]; };
layout(set = 0, binding = 4) readonly buffer TopkIdx_   { int topk_idx[]; };

layout(push_constant) uniform Push {
    uint numExperts;
    uint pairs;
    uint k;
} pc;

shared int sExpert[WGSIZE];
shared uint sRank[WGSIZE];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint e = pc.numExperts;
    uint pairs = pc.pairs;

    for (uint i = tid; i < e; i += WGSIZE) counts[i] = 0u;
    memoryBarrierBuffer();
    barrier();

    for (uint p = tid; p < pairs; p += WGSIZE) {
        int x = topk_idx[p];
        if (uint(x) < e) atomicAdd(counts[x], 1u);
    }
    memoryBarrierBuffer();
    barrier();

    if (tid == 0u) {
        uint total = 0u;
        for (uint i = 0u; i < e; i++) {
            offsets[i] = total;
            total += counts[i];
        }
        offsets[e] = total;
    }
    memoryBarrierBuffer();
    barrier();

    uint total = offsets[e];
    for (uint p = total + tid; p < pairs; p += WGSIZE) permuted_token[p] = -1;
    // Counts double as the running placement cursor; each expert ends back at its own total.
    for (uint i = tid; i < e; i += WGSIZE) counts[i] = 0u;
    memoryBarrierBuffer();
    barrier();

    for (uint base = 0u; base < pairs; base += WGSIZE) {
        uint p = base + tid;
        int x = -1;
        if (p < pairs) x = topk_idx[p];
        sExpert[tid] = x;
        barrier();
        bool valid = p < pairs && uint(x) < e;
        uint rank = 0u;
        bool last = true;
        if (valid) {
            for (uint j = 0u; j < tid; j++) rank += (sExpert[j] == x) ? 1u : 0u;
            for (uint j = tid + 1u; j < WGSIZE && base + j < pairs; j++) {
                if (sExpert[j] == x) { last = false; break; }
            }
            uint slot = offsets[x] + counts[x] + rank;
            permuted_token[slot] = int(p / pc.k);
            pair_slot[p] = int(slot);
            sRank[tid] = rank + 1u;
        } else if (p < pairs) {
            pair_slot[p] = -1;
        }
        memoryBarrierBuffer();
        barrier();
        if (valid && last) counts[x] += sRank[tid];
        memoryBarrierBuffer();
        barrier();
    }
}
