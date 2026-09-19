// layernorm_modulate: per-row LayerNorm with NO learned affine, then a PER-BATCH modulation.
//   y = (x - mean) * rsqrt(var + eps) * (1 + scale[b]) + shift[b]
// where b = row / seqLen, so every row of a sequence shares one modulation vector.
//
// The (1 + scale) convention is the DiT one and is load-bearing: scale is a residual around identity,
// so a zero modulation must leave the normalized value untouched rather than zero it.
//
// One workgroup per row. fp32 accumulator regardless of storage dtype.
//
// Bindings: 0=x (in), 1=scale (in, per-batch x dim), 2=shift (in, per-batch x dim), 3=y (out)
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

layout(set = 0, binding = 0) readonly  buffer X_  { DTYPE x[]; };
layout(set = 0, binding = 1) readonly  buffer SC_ { float scaleT[]; };
layout(set = 0, binding = 2) readonly  buffer SH_ { float shiftT[]; };
layout(set = 0, binding = 3) writeonly buffer Y_  { DTYPE y[]; };

layout(push_constant) uniform Push {
    uint normDim;     // size of last (normed) dim
    uint totalRows;   // count of rows = elements / normDim
    uint seqLen;      // rows per batch entry; the modulation index is row / seqLen
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
        // e.g. Intel subgroup 8 at local 256 -> 32 subgroups).
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
    uint modOff = (row / pc.seqLen) * pc.normDim;
    for (uint i = gl_LocalInvocationIndex; i < pc.normDim; i += gl_WorkGroupSize.x) {
        float v = (TO_F32(x[baseOff + i]) - mean) * invStd;
        y[baseOff + i] = FROM_F32(v * (1.0 + scaleT[modOff + i]) + shiftT[modOff + i]);
    }
}
