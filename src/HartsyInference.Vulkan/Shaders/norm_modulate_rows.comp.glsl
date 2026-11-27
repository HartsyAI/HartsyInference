// norm_modulate_rows: one row along the last axis per workgroup, y = (norm(x) · w + b) · scale + shift with every absent term
// dropped (IBackend.NormModulate). KIND picks the norm (0 none, 1 LayerNorm, 2 RMSNorm, 3 L2-floored RMS), ONE_PLUS the scale
// mode, HAS_* which vectors are read; the unread bindings are placeholders. Row r reads modulation row rowIndex[r] (HAS_INDEX)
// or r / rowsPerMod (every row reads row 0 when rowsPerMod is 0), at scaleOffset / shiftOffset + m · rowStride. IN_FP16 and
// OUT_FP16 pick the activation dtypes independently. Matches NormModulateReference up to the reduction order.
//
// Bindings: 0=x, 1=weight [width], 2=bias [width], 3=scale, 4=shift, 5=rowIndex (i32), 6=y
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#ifndef IN_FP16
#define IN_FP16 0
#endif
#ifndef OUT_FP16
#define OUT_FP16 0
#endif
#if IN_FP16 == 1 || OUT_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#endif
#if IN_FP16 == 1
#define IN_T float16_t
#else
#define IN_T float
#endif
#if OUT_FP16 == 1
#define OUT_T float16_t
#else
#define OUT_T float
#endif

layout(local_size_x_id = 0) in;
layout(constant_id = 3) const uint KIND = 0u;
layout(constant_id = 4) const bool ONE_PLUS = false;
layout(constant_id = 5) const bool HAS_WEIGHT = false;
layout(constant_id = 6) const bool HAS_BIAS = false;
layout(constant_id = 7) const bool HAS_SCALE = false;
layout(constant_id = 8) const bool HAS_SHIFT = false;
layout(constant_id = 9) const bool HAS_INDEX = false;

layout(set = 0, binding = 0) readonly  buffer X_  { IN_T x[]; };
layout(set = 0, binding = 1) readonly  buffer W_  { float weight[]; };
layout(set = 0, binding = 2) readonly  buffer B_  { float bias[]; };
layout(set = 0, binding = 3) readonly  buffer Sc_ { float scale[]; };
layout(set = 0, binding = 4) readonly  buffer Sh_ { float shift[]; };
layout(set = 0, binding = 5) readonly  buffer Ix_ { int rowIndex[]; };
layout(set = 0, binding = 6) writeonly buffer Y_  { OUT_T y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint width;
    uint rowsPerMod;
    uint rowStride;
    uint scaleOffset;
    uint shiftOffset;
    float eps;
} pc;

shared float warpSums[64];
shared float gMean;
shared float gFactor;

float block_sum(float v) {
    v = subgroupAdd(v);
    if (subgroupElect()) warpSums[gl_SubgroupID] = v;
    barrier();
    float total = 0.0;
    if (gl_SubgroupID == 0u) {
        float w = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) w += warpSums[k];
        total = subgroupAdd(w);
    }
    barrier();
    return total;
}

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint base = row * pc.width;

    float mean = 0.0, factor = 1.0;
    if (KIND == 1u) {
        float partial = 0.0;
        for (uint i = gl_LocalInvocationIndex; i < pc.width; i += gl_WorkGroupSize.x) partial += float(x[base + i]);
        float sum = block_sum(partial);
        if (gl_LocalInvocationIndex == 0u) gMean = sum / float(pc.width);
        barrier();
        mean = gMean;
        float vpart = 0.0;
        for (uint i = gl_LocalInvocationIndex; i < pc.width; i += gl_WorkGroupSize.x) {
            float d = float(x[base + i]) - mean;
            vpart += d * d;
        }
        float variance = block_sum(vpart);
        if (gl_LocalInvocationIndex == 0u) gFactor = inversesqrt(variance / float(pc.width) + pc.eps);
        barrier();
        factor = gFactor;
    } else if (KIND != 0u) {
        float partial = 0.0;
        for (uint i = gl_LocalInvocationIndex; i < pc.width; i += gl_WorkGroupSize.x) {
            float v = float(x[base + i]);
            partial += v * v;
        }
        float sumSq = block_sum(partial);
        if (gl_LocalInvocationIndex == 0u)
            gFactor = KIND == 2u ? inversesqrt(sumSq / float(pc.width) + pc.eps) : sqrt(float(pc.width)) / max(sqrt(sumSq), pc.eps);
        barrier();
        factor = gFactor;
    }

    uint m = HAS_INDEX ? uint(rowIndex[row]) : (pc.rowsPerMod == 0u ? 0u : row / pc.rowsPerMod);
    uint scaleAt = pc.scaleOffset + m * pc.rowStride;
    uint shiftAt = pc.shiftOffset + m * pc.rowStride;
    for (uint i = gl_LocalInvocationIndex; i < pc.width; i += gl_WorkGroupSize.x) {
        float v = float(x[base + i]);
        if (KIND == 1u) v = (v - mean) * factor;
        else if (KIND != 0u) v *= factor;
        if (HAS_WEIGHT) v *= weight[i];
        if (HAS_BIAS) v += bias[i];
        if (HAS_SCALE) v *= ONE_PLUS ? 1.0 + scale[scaleAt + i] : scale[scaleAt + i];
        if (HAS_SHIFT) v += shift[shiftAt + i];
        y[base + i] = OUT_T(v);
    }
}
