// softmax: per-row online (max-subtract) softmax for numerical stability.
// One workgroup per row.
//
// Bindings: 0=x (in), 1=y (out)
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#define TO_F32(x) float(x)
#define FROM_F32(x) float16_t(x)
#else
#define DTYPE float
#define TO_F32(x) (x)
#define FROM_F32(x) (x)
#endif

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_ { DTYPE x[]; };
layout(set = 0, binding = 1) writeonly buffer Y_ { DTYPE y[]; };

layout(push_constant) uniform Push {
    uint N;        // normalize-along dim (last)
    uint totalRows;
    uint baseOffset;   // element offset for batched dispatches
} pc;

shared float warp_max[32];
shared float warp_sum[32];
shared float gMax;
shared float gSum;

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.totalRows) return;
    uint base = pc.baseOffset + row * pc.N;

    // Pass 1: row-max
    float maxv = -3.4e38;
    for (uint i = gl_LocalInvocationIndex; i < pc.N; i += gl_WorkGroupSize.x)
        maxv = max(maxv, TO_F32(x[base + i]));
    maxv = subgroupMax(maxv);
    if (subgroupElect()) warp_max[gl_SubgroupID] = maxv;
    barrier();
    if (gl_SubgroupID == 0u) {
        float v = (gl_SubgroupInvocationID < gl_NumSubgroups) ? warp_max[gl_SubgroupInvocationID] : -3.4e38;
        v = subgroupMax(v);
        if (subgroupElect()) gMax = v;
    }
    barrier();

    // Pass 2: row sum of exp(x - max)
    float maxF = gMax;
    float sum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.N; i += gl_WorkGroupSize.x)
        sum += exp(TO_F32(x[base + i]) - maxF);
    sum = subgroupAdd(sum);
    if (subgroupElect()) warp_sum[gl_SubgroupID] = sum;
    barrier();
    if (gl_SubgroupID == 0u) {
        float v = (gl_SubgroupInvocationID < gl_NumSubgroups) ? warp_sum[gl_SubgroupInvocationID] : 0.0;
        v = subgroupAdd(v);
        if (subgroupElect()) gSum = v;
    }
    barrier();

    // Pass 3: write normalized
    float inv = 1.0 / gSum;
    for (uint i = gl_LocalInvocationIndex; i < pc.N; i += gl_WorkGroupSize.x) {
        float v = exp(TO_F32(x[base + i]) - maxF) * inv;
        y[base + i] = FROM_F32(v);
    }
}
