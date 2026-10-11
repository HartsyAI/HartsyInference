// Fused dequantizing GEMV for the decode-step Linear: out[m, n] = sum_k x[m, k] * W[n, k] (+ bias[n]) for M <= 8 activation
// rows, reading the GGUF weight blocks (or an F32 matrix) directly instead of a dequantized copy. Built once per weight format by
// build.sh, one module per QTYPE: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = Q4_0, 4 = Q5_0, 5 = Q5_K, 6 = F32 (gemv_quant_<type>). The
// per-unit decoders live in gemv_quant_unit.glsl, shared with the MoE expert shaders.
//
// One workgroup per output row. The row is cut into work units of a few words of quants (Q8_0: 4 values, 8 units per
// block; Q4_K: 16 bytes of nibbles = 32 values, 8 units per super-block; Q6_K: 4 l-positions of one half = 16 values, 16
// units per super-block) and lane t takes units t, t + WG, ..., so a subgroup reads one contiguous run of weight bytes and
// one contiguous run of activations (vec4 loads) per step. Giving each lane a whole block instead strides the activation
// loads by a block per lane and measured at about 50 GB/s on an RTX 3060.
// Plain float math: each unit dequantizes in registers and accumulates its dot per activation row; the row total is a
// subgroup sum followed by a shared-memory sum across subgroups. The integer dot-product variant (dotPacked4x8 against an
// int8 activation, as the CUDA dp4a kernels do) is a follow-up; the distro glslang cannot compile it.

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

// N output rows, K inputs per row (a multiple of the block).
layout(push_constant) uniform Push { uint N; uint K; } pc;

shared float partials[M * 32u];

float acc[M];

#define WBUF w_data
#define FN_READ32 readU32
#define FN_READ16 readU16
#define FN_HALF halfAt
#define FN_UNIT accumulateUnit
#define ACC acc
#include "gemv_quant_unit.glsl"

void main() {
    // A vocabulary-sized N exceeds the 65535 workgroups a dimension is guaranteed, so the grid may be two-dimensional.
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= pc.N) return;
    uint lane = gl_LocalInvocationID.x;
    uint wg = gl_WorkGroupSize.x;
    uint blocksPerRow = pc.K / BLOCK_ELEMS;
    uint units = blocksPerRow * UNITS_PER_BLOCK;
    uint rowBase = row * blocksPerRow * BLOCK_BYTES;
    uint K4 = pc.K >> 2;

    for (uint m = 0u; m < M; m++) acc[m] = 0.0;

    for (uint u = lane; u < units; u += wg) accumulateUnit(u, rowBase, K4, 0u);

    // Row total: subgroup sum, then one partial per subgroup through shared memory.
    uint sgCount = (wg + gl_SubgroupSize - 1u) / gl_SubgroupSize;
    for (uint m = 0u; m < M; m++) {
        float s = subgroupAdd(acc[m]);
        if (subgroupElect()) partials[m * 32u + gl_SubgroupID] = s;
    }
    barrier();
    if (lane < M) {
        float total = 0.0;
        for (uint g = 0u; g < sgCount; g++) total += partials[lane * 32u + g];
        if (HAS_BIAS) total += bias_data[row];
        out_data[lane * pc.N + row] = total;
    }
}
