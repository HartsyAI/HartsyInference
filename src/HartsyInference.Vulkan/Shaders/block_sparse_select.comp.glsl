// 3D block-sparse attention, stage 2 of 3 (selection). BlockSparseAttentionReference in Core defines it;
// block_sparse_attention.cu is the CUDA form.
//
// Per (query block, head, batch): the `chosen` >= 1 key blocks with the highest dot product of the two block means,
// ascending, [B, H, queryBlocks, chosen]. The scores are `precise` and summed over channels ascending, the reference's
// operations. Each score becomes an unsigned key that orders as the reference ranks (NaN above every number, -0 equal to
// +0); the threshold is the chosen-th largest key, found a bit at a time; every block above it is read, and as many of
// the blocks equal to it as are still wanted, lowest index first.
#version 460
#extension GL_GOOGLE_include_directive : require

#define MAX_DIM 128
// With the mean and the counts, the keys fill what a 32 KiB workgroup holds: VulkanBackend.BlockSparseMaxKeyBlocks.
#define MAX_KEY_BLOCKS 7680

layout(local_size_x = 128) in;

layout(set = 0, binding = 0) writeonly buffer S_ { int selected[]; };
layout(set = 0, binding = 1) readonly  buffer Q_ { float query_means[]; };
layout(set = 0, binding = 2) readonly  buffer K_ { float key_means[]; };

layout(push_constant) uniform Push {
    uint dim;
    uint heads;
    uint queryBlocks;
    uint keyBlocks;
    uint chosen;
    uint firstBlock;
} pc;

shared float s_mean[MAX_DIM];
shared uint s_keys[MAX_KEY_BLOCKS];
shared uint s_counts[128];

#include "block_sparse_common.glsl"

uint rankKey(float score) {
    if (bsaIsNaN(score)) return 0xFFFFFFFFu;
    if (bsaIsZero(score)) return 0x80000000u;
    uint bits = floatBitsToUint(score);
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
}

void main() {
    uint qb = pc.firstBlock + gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint tid = gl_LocalInvocationID.x;
    uint head = b * pc.heads + h;

    for (uint d = tid; d < pc.dim; d += 128u) s_mean[d] = query_means[(head * pc.queryBlocks + qb) * pc.dim + d];
    barrier();
    for (uint n = tid; n < pc.keyBlocks; n += 128u) {
        uint k = (head * pc.keyBlocks + n) * pc.dim;
        precise float dot = 0.0;
        for (uint d = 0u; d < pc.dim; ++d) {
            precise float term = s_mean[d] * key_means[k + d];
            dot = dot + term;
        }
        s_keys[n] = rankKey(dot);
    }
    barrier();

    uint threshold = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        uint candidate = threshold | (1u << uint(bit));
        uint count = 0u;
        for (uint n = tid; n < pc.keyBlocks; n += 128u) count += s_keys[n] >= candidate ? 1u : 0u;
        s_counts[tid] = count;
        barrier();
        uint total = 0u;
        for (uint i = 0u; i < 128u; ++i) total += s_counts[i];
        barrier();
        if (total >= pc.chosen) threshold = candidate;
    }

    if (tid != 0u) return;
    uint above = 0u;
    for (uint n = 0u; n < pc.keyBlocks; ++n) above += s_keys[n] > threshold ? 1u : 0u;
    uint ties = pc.chosen - above;
    uint at = (head * pc.queryBlocks + qb) * pc.chosen;
    uint taken = 0u;
    for (uint n = 0u; n < pc.keyBlocks && taken < pc.chosen; ++n) {
        bool take = s_keys[n] > threshold;
        if (!take && s_keys[n] == threshold && ties > 0u) {
            take = true;
            --ties;
        }
        if (take) {
            selected[at + taken] = int(n);
            ++taken;
        }
    }
}
