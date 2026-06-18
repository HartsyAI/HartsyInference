// groupnorm: per-(batch, group) GroupNorm with scale + bias.
//   y = (x - mean) * rsqrt(var + eps) * weight[c] + bias[c]
//
// One workgroup per (batch * group). Mean / variance accumulated in fp32 with
// subgroupAdd + cross-subgroup shared-memory reduction. Inputs in NCHW layout.
//
// Bindings: 0=x (in), 1=weight (per-channel, fp32), 2=bias (per-channel, fp32), 3=y (out)
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
    uint N;        // batch
    uint C;        // total channels
    uint H;        // spatial height
    uint W;        // spatial width
    uint groups;   // number of GN groups
    float eps;
} pc;

shared float warp_sum[64];
shared float warp_sqsum[64];
shared float gMean;
shared float gInvStd;

void main() {
    uint nGroup    = gl_WorkGroupID.x;
    uint b         = nGroup / pc.groups;
    uint g         = nGroup % pc.groups;
    uint cPerGroup = pc.C / pc.groups;
    uint groupSize = cPerGroup * pc.H * pc.W;
    uint baseOff   = (b * pc.C + g * cPerGroup) * pc.H * pc.W;

    // Phase 1: per-thread sum + sqsum
    float sum = 0.0;
    float sqsum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < groupSize; i += gl_WorkGroupSize.x) {
        float v = TO_F32(x[baseOff + i]);
        sum   += v;
        sqsum += v * v;
    }

    // Phase 2: subgroup reduction
    sum   = subgroupAdd(sum);
    sqsum = subgroupAdd(sqsum);

    // Phase 3: cross-subgroup via shared mem (one slot per subgroup)
    if (subgroupElect()) {
        warp_sum[gl_SubgroupID] = sum;
        warp_sqsum[gl_SubgroupID] = sqsum;
    }
    barrier();

    if (gl_SubgroupID == 0u) {
        // Strided fold so gl_NumSubgroups > gl_SubgroupSize is handled (small-subgroup GPUs).
        float w = 0.0, w2 = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) {
            w  += warp_sum[k];
            w2 += warp_sqsum[k];
        }
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

    // Phase 4: normalize + affine
    float mean   = gMean;
    float invStd = gInvStd;
    for (uint i = gl_LocalInvocationIndex; i < groupSize; i += gl_WorkGroupSize.x) {
        uint cIdx = (i / (pc.H * pc.W)) + g * cPerGroup;
        float v = (TO_F32(x[baseOff + i]) - mean) * invStd;
        v = v * weight[cIdx] + bias[cIdx];
        y[baseOff + i] = FROM_F32(v);
    }
}
