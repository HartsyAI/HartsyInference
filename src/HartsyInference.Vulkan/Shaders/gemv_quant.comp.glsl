// Fused dequantizing GEMV for the decode-step Linear: out[m, n] = sum_k x[m, k] * W[n, k] (+ bias[n]) for M <= 16 activation
// rows, reading the GGUF weight blocks (or an F32 matrix) directly instead of a dequantized copy. Built once per weight format by
// build.sh, one module per QTYPE: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = Q4_0, 4 = Q5_0, 5 = Q5_K, 6 = F32 (gemv_quant_<type>). The
// per-unit decoders live in gemv_quant_unit.glsl, shared with the MoE expert shaders.
//
// The row is cut into work units of a few words of quants (Q8_0: 4 values, 8 units per block; Q4_K: 16 bytes of nibbles = 32
// values, 8 units per super-block; Q6_K: 4 l-positions of one half = 16 values, 16 units per super-block). A row is walked by
// SPLIT subgroups: subgroup part p of the row takes units p * subgroupSize + lane, stepping by SPLIT * subgroupSize, so one
// subgroup step reads one contiguous run of weight bytes and one of activations (vec4 loads).
//   SPLIT = 1: one subgroup per row. The row total is one subgroupAdd per activation row; no shared memory, no barrier. The
//              workgroup holds ROWS_PER_WG rows (its subgroups). Best where a lane's loads are the cost (memory-bound formats).
//   SPLIT > 1: SPLIT subgroups share a row and sum through shared memory, with one barrier. Best where a unit's decode is the
//              cost (Q5_0's bit unpacking), since more lanes work on one row.
// A subgroup-per-row lane unrolls four units per step, so four units' loads are in flight rather than each waited on in turn;
// a split row takes one unit per step (see the main loop).
// Plain float math: each unit dequantizes in registers and accumulates its dot per activation row. The integer dot-product
// variant (dotPacked4x8 against an int8 activation, as the CUDA dp4a kernels do) is a follow-up; the distro glslang cannot
// compile it.

#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer W_    { uint w_data[]; };
layout(set = 0, binding = 1) readonly  buffer X_    { vec4 x4[]; };
layout(set = 0, binding = 2) writeonly buffer Out_  { float out_data[]; };
layout(set = 0, binding = 3) readonly  buffer Bias_ { float bias_data[]; };

layout(constant_id = 1) const bool HAS_BIAS = false;
// Activation rows. A specialization constant, not a push constant, so every per-row loop has a fixed trip count and the
// accumulators stay in registers; indexed by a runtime bound they spill to local memory.
layout(constant_id = 2) const uint M = 1u;
// Subgroups per output row (see the header). A specialization constant; the host divides the workgroup's subgroups by it.
layout(constant_id = 3) const uint SPLIT = 1u;
// Output rows per workgroup: the workgroup's subgroups divided by SPLIT, computed by the host (QuantGemv.cs) and not re-derived here.
layout(constant_id = 4) const uint ROWS_PER_WG = 1u;

// N output rows, K inputs per row (a multiple of the block).
layout(push_constant) uniform Push { uint N; uint K; } pc;

float acc[M];

// Row partials for SPLIT > 1: one slot per (row in the workgroup, activation row, subgroup of the row). A workgroup holds at most
// 32 subgroups.
shared float partials[32u * M];

#define WBUF w_data
#define FN_READ32 readU32
#define FN_READ16 readU16
#define FN_HALF halfAt
#define FN_UNIT accumulateUnit
#define ACC acc
#include "gemv_quant_unit.glsl"

void main() {
    uint sg = gl_SubgroupSize;
    uint rowInWg = gl_SubgroupID / SPLIT;
    uint part = gl_SubgroupID % SPLIT;
    // The grid holds ceil(N / ROWS_PER_WG) workgroups. A vocabulary-sized N exceeds the 65535 workgroups a dimension is
    // guaranteed, so the grid may be two-dimensional.
    uint row = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * ROWS_PER_WG + rowInWg;
    bool live = row < pc.N;
    uint lane = gl_SubgroupInvocationID;
    uint blocksPerRow = pc.K / BLOCK_ELEMS;
    uint units = blocksPerRow * UNITS_PER_BLOCK;
    uint rowBase = row * blocksPerRow * BLOCK_BYTES;   // read only when the row is live
    uint K4 = pc.K >> 2;
    uint stride = SPLIT * sg;

    for (uint m = 0u; m < M; m++) acc[m] = 0.0;

    if (live) {
        uint u = part * sg + lane;
        // Four units per step for a subgroup per row (SPLIT 1). A split row takes one unit per step: its unrolled form measured
        // slower on Q5_0, the split format, in the decode profile.
        if (SPLIT == 1u) {
            for (; u + 3u * stride < units; u += 4u * stride) {
                accumulateUnit(u, rowBase, K4, 0u);
                accumulateUnit(u + stride, rowBase, K4, 0u);
                accumulateUnit(u + 2u * stride, rowBase, K4, 0u);
                accumulateUnit(u + 3u * stride, rowBase, K4, 0u);
            }
        }
        for (; u < units; u += stride) accumulateUnit(u, rowBase, K4, 0u);
    }

    if (SPLIT == 1u) {
        // One subgroup per row: the row total is one subgroup sum per activation row.
        for (uint m = 0u; m < M; m++) {
            float total = subgroupAdd(acc[m]);
            if (live && subgroupElect()) {
                if (HAS_BIAS) total += bias_data[row];
                out_data[m * pc.N + row] = total;
            }
        }
        return;
    }

    // SPLIT > 1: each subgroup's sum goes to its slot; after the barrier, the row's first subgroup adds the slots. Every
    // invocation reaches the barrier (no early return above it), so the rows of a workgroup can be dead or live independently.
    for (uint m = 0u; m < M; m++) {
        float s = subgroupAdd(acc[m]);
        if (subgroupElect()) partials[(rowInWg * M + m) * SPLIT + part] = s;
    }
    barrier();
    if (live && part == 0u) {
        for (uint m = lane; m < M; m += sg) {
            float total = 0.0;
            for (uint p = 0u; p < SPLIT; p++) total += partials[(rowInWg * M + m) * SPLIT + p];
            if (HAS_BIAS) total += bias_data[row];
            out_data[m * pc.N + row] = total;
        }
    }
}
