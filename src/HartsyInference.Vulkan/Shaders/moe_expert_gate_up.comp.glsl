// Expert-indexed gate and up projections of a MoE layer's routed pairs, with the activation fused:
//   act[pair, n] = act(gate[e] . x[token]) * (up[e] . x[token]),   e = topkIdx[pair],  token = pair / topk
// for decode and a short speculative verify (at most 16 tokens, so at most 16 * topk pairs). The expert id is read from device
// memory, so the layer's routed stage runs from the router to the combine with no host read. Built once per weight format by
// build.sh, one module per QTYPE (gemv_quant_unit.glsl: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = Q4_0, 4 = Q5_0, 5 = Q5_K).
//
// The gate and up stacks are each one buffer holding the experts back to back, strideBytes apart. One SUBGROUP computes one
// output row of one pair: its lanes take the row's work units lane, lane + subgroupSize, ... (the step-1 inner loop, shared
// through the include), so a subgroup reads one contiguous run of the row; a workgroup is several subgroups, i.e. several rows,
// and the second grid axis is the pair. Gate and up are accumulated in the same loop so the activation reads are shared.
// One activation row per workgroup, so the accumulator counts are constants (M = 1) and stay in registers.

#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer Gate_ { uint gate_data[]; };
layout(set = 0, binding = 1) readonly  buffer Up_   { uint up_data[]; };
layout(set = 0, binding = 2) readonly  buffer X_    { vec4 x4[]; };
layout(set = 0, binding = 3) readonly  buffer Idx_  { int topk_idx[]; };
layout(set = 0, binding = 4) writeonly buffer Act_  { float act_data[]; };

// tanh-GELU instead of SiLU (Gemma-4's GeGLU experts).
layout(constant_id = 1) const bool GELU = false;

// N output rows per expert, K inputs per row, topk slots per token; pairs [firstPair, firstPair + the grid's y extent) are
// computed (a launcher chunks the pair axis by the 65535 workgroups a dimension is guaranteed); strideBytes between experts of a
// stack; numExperts bounds the id read from the device (an id outside it writes zeros rather than reading past the stack).
layout(push_constant) uniform Push { uint N; uint K; uint topk; uint firstPair; uint strideBytes; uint numExperts; } pc;

const uint M = 1u;
float accG[M];
float accU[M];

#define WBUF gate_data
#define FN_READ32 gateRead32
#define FN_READ16 gateRead16
#define FN_HALF gateHalf
#define FN_UNIT gateUnit
#define ACC accG
#include "gemv_quant_unit.glsl"
#undef WBUF
#undef FN_READ32
#undef FN_READ16
#undef FN_HALF
#undef FN_UNIT
#undef ACC

#define WBUF up_data
#define FN_READ32 upRead32
#define FN_READ16 upRead16
#define FN_HALF upHalf
#define FN_UNIT upUnit
#define ACC accU
#include "gemv_quant_unit.glsl"

float siluf(float x) { return x / (1.0 + exp(-x)); }
float geluTanh(float x) { return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }

void main() {
    uint row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID;
    if (row >= pc.N) return;
    uint pair = pc.firstPair + gl_WorkGroupID.y;
    int e = topk_idx[pair];
    if (e < 0 || uint(e) >= pc.numExperts) {
        if (subgroupElect()) act_data[pair * pc.N + row] = 0.0;
        return;
    }
    uint blocksPerRow = pc.K / BLOCK_ELEMS;
    uint units = blocksPerRow * UNITS_PER_BLOCK;
    uint rowBase = uint(e) * pc.strideBytes + row * blocksPerRow * BLOCK_BYTES;
    uint K4 = pc.K >> 2;
    uint xOff = (pair / pc.topk) * K4;

    accG[0] = 0.0;
    accU[0] = 0.0;
    for (uint u = gl_SubgroupInvocationID; u < units; u += gl_SubgroupSize) {
        gateUnit(u, rowBase, K4, xOff);
        upUnit(u, rowBase, K4, xOff);
    }
    float g = subgroupAdd(accG[0]);
    float up = subgroupAdd(accU[0]);
    if (subgroupElect()) act_data[pair * pc.N + row] = (GELU ? geluTanh(g) : siluf(g)) * up;
}
