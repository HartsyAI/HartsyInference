// Expert-indexed down projection of a MoE layer's routed pairs:  slotOut[pair, n] = down[e] . act[pair],  e = topkIdx[pair].
// The companion of moe_expert_gate_up: same subgroup-per-output-row layout and the same shared inner loop
// (gemv_quant_unit.glsl, QTYPE as in gemv_quant). The down stack is one buffer holding the experts back to back, strideBytes
// apart; its type may differ from the gate/up stacks' (OLMoE: Q4_K gate/up, Q6_K down), which is why it is its own module.

#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer Down_ { uint down_data[]; };
layout(set = 0, binding = 1) readonly  buffer X_    { vec4 x4[]; };
layout(set = 0, binding = 2) readonly  buffer Idx_  { int topk_idx[]; };
layout(set = 0, binding = 3) writeonly buffer Out_  { float out_data[]; };

// N output rows (the hidden size), K inputs per row (the expert intermediate size); pairs [firstPair, firstPair + the grid's y
// extent); strideBytes between experts; numExperts bounds the id read from the device.
layout(push_constant) uniform Push { uint N; uint K; uint firstPair; uint strideBytes; uint numExperts; } pc;

const uint M = 1u;
float acc[M];

#define WBUF down_data
#define FN_READ32 readU32
#define FN_READ16 readU16
#define FN_HALF halfAt
#define FN_UNIT accumulateUnit
#define ACC acc
#include "gemv_quant_unit.glsl"

void main() {
    uint row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID;
    if (row >= pc.N) return;
    uint pair = pc.firstPair + gl_WorkGroupID.y;
    int e = topk_idx[pair];
    if (e < 0 || uint(e) >= pc.numExperts) {
        if (subgroupElect()) out_data[pair * pc.N + row] = 0.0;
        return;
    }
    uint blocksPerRow = pc.K / BLOCK_ELEMS;
    uint units = blocksPerRow * UNITS_PER_BLOCK;
    uint rowBase = uint(e) * pc.strideBytes + row * blocksPerRow * BLOCK_BYTES;
    uint K4 = pc.K >> 2;
    uint xOff = pair * K4;

    acc[0] = 0.0;
    for (uint u = gl_SubgroupInvocationID; u < units; u += gl_SubgroupSize) accumulateUnit(u, rowBase, K4, xOff);
    float total = subgroupAdd(acc[0]);
    if (subgroupElect()) out_data[pair * pc.N + row] = total;
}
