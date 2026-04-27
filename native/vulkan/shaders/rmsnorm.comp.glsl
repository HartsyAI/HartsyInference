// rmsnorm: per-row RMS normalization (no mean subtraction, no bias).
//   y = x * rsqrt(mean(x^2) + eps) * weight
// Used by T5 / Flux / DiT-family models.
// One workgroup per row.
//
// Bindings: 0=x (in), 1=weight (per-dim, fp32), 2=y (out)
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
layout(set = 0, binding = 1) readonly  buffer W_ { float weight[]; };
layout(set = 0, binding = 2) writeonly buffer Y_ { DTYPE y[]; };

layout(push_constant) uniform Push {
    uint normDim;
    uint totalRows;
    float eps;
} pc;

shared float warp_sqsum[32];
shared float gInvStd;

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.totalRows) return;
    uint baseOff = row * pc.normDim;

    float sqsum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.normDim; i += gl_WorkGroupSize.x) {
        float v = TO_F32(x[baseOff + i]);
        sqsum += v * v;
    }
    sqsum = subgroupAdd(sqsum);
    if (subgroupElect()) warp_sqsum[gl_SubgroupID] = sqsum;
    barrier();

    if (gl_SubgroupID == 0u) {
        float w = (gl_SubgroupInvocationID < gl_NumSubgroups) ? warp_sqsum[gl_SubgroupInvocationID] : 0.0;
        w = subgroupAdd(w);
        if (subgroupElect()) gInvStd = inversesqrt(w / float(pc.normDim) + pc.eps);
    }
    barrier();

    float invStd = gInvStd;
    for (uint i = gl_LocalInvocationIndex; i < pc.normDim; i += gl_WorkGroupSize.x) {
        float v = TO_F32(x[baseOff + i]) * invStd * weight[i];
        y[baseOff + i] = FROM_F32(v);
    }
}
