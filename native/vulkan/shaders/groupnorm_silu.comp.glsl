// groupnorm_silu: fused groupnorm + SiLU. See groupnorm.comp.glsl for the math.
// Eliminates one full-tensor write+read between the norm and the activation.
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
layout(set = 0, binding = 2) readonly  buffer B_ { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer Y_ { DTYPE y[]; };

layout(push_constant) uniform Push {
    uint N;
    uint C;
    uint H;
    uint W;
    uint groups;
    float eps;
} pc;

shared float warp_sum[32];
shared float warp_sqsum[32];
shared float gMean;
shared float gInvStd;

void main() {
    uint nGroup    = gl_WorkGroupID.x;
    uint b         = nGroup / pc.groups;
    uint g         = nGroup % pc.groups;
    uint cPerGroup = pc.C / pc.groups;
    uint groupSize = cPerGroup * pc.H * pc.W;
    uint baseOff   = (b * pc.C + g * cPerGroup) * pc.H * pc.W;

    float sum = 0.0;
    float sqsum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < groupSize; i += gl_WorkGroupSize.x) {
        float v = TO_F32(x[baseOff + i]);
        sum   += v;
        sqsum += v * v;
    }

    sum   = subgroupAdd(sum);
    sqsum = subgroupAdd(sqsum);

    if (subgroupElect()) {
        warp_sum[gl_SubgroupID] = sum;
        warp_sqsum[gl_SubgroupID] = sqsum;
    }
    barrier();

    if (gl_SubgroupID == 0u) {
        float w  = (gl_SubgroupInvocationID < gl_NumSubgroups) ? warp_sum[gl_SubgroupInvocationID]   : 0.0;
        float w2 = (gl_SubgroupInvocationID < gl_NumSubgroups) ? warp_sqsum[gl_SubgroupInvocationID] : 0.0;
        w  = subgroupAdd(w);
        w2 = subgroupAdd(w2);
        if (subgroupElect()) {
            float invN = 1.0 / float(groupSize);
            float mean = w * invN;
            float var  = w2 * invN - mean * mean;
            gMean   = mean;
            gInvStd = inversesqrt(var + pc.eps);
        }
    }
    barrier();

    float mean   = gMean;
    float invStd = gInvStd;
    for (uint i = gl_LocalInvocationIndex; i < groupSize; i += gl_WorkGroupSize.x) {
        uint cIdx = (i / (pc.H * pc.W)) + g * cPerGroup;
        float v = (TO_F32(x[baseOff + i]) - mean) * invStd;
        v = v * weight[cIdx] + bias[cIdx];
        // Fused SiLU: x * sigmoid(x)
        v = v / (1.0 + exp(-v));
        y[baseOff + i] = FROM_F32(v);
    }
}
