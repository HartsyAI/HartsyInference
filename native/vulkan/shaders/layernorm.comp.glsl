// layernorm: per-row LayerNorm with per-feature scale + bias.
//   y = (x - mean) * rsqrt(var + eps) * weight + bias
// One workgroup per row. fp32 accumulator.
//
// Bindings: 0=x (in), 1=weight (per-dim, fp32), 2=bias (per-dim, fp32), 3=y (out)
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
    uint normDim;     // size of last (normed) dim
    uint totalRows;   // count of rows = elements / normDim
    float eps;
} pc;

shared float warp_sum[64];
shared float warp_sqsum[64];
shared float gMean;
shared float gInvStd;

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.totalRows) return;
    uint baseOff = row * pc.normDim;

    float sum = 0.0, sqsum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.normDim; i += gl_WorkGroupSize.x) {
        float v = TO_F32(x[baseOff + i]);
        sum += v; sqsum += v * v;
    }

    sum   = subgroupAdd(sum);
    sqsum = subgroupAdd(sqsum);
    if (subgroupElect()) { warp_sum[gl_SubgroupID] = sum; warp_sqsum[gl_SubgroupID] = sqsum; }
    barrier();

    if (gl_SubgroupID == 0u) {
        // Strided fold so gl_NumSubgroups > gl_SubgroupSize is handled (small-subgroup GPUs,
        // e.g. Intel subgroup 8 at local 256 -> 32 subgroups). Each lane sums its strided
        // share of the partials, then one subgroup reduction combines the lanes.
        float w = 0.0, w2 = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) {
            w  += warp_sum[k];
            w2 += warp_sqsum[k];
        }
        w  = subgroupAdd(w);
        w2 = subgroupAdd(w2);
        if (subgroupElect()) {
            float invN = 1.0 / float(pc.normDim);
            float mean = w * invN;
            float var  = w2 * invN - mean * mean;
            gMean   = mean;
            gInvStd = inversesqrt(var + pc.eps);
        }
    }
    barrier();

    float mean = gMean, invStd = gInvStd;
    for (uint i = gl_LocalInvocationIndex; i < pc.normDim; i += gl_WorkGroupSize.x) {
        float v = (TO_F32(x[baseOff + i]) - mean) * invStd;
        v = v * weight[i] + bias[i];
        y[baseOff + i] = FROM_F32(v);
    }
}
