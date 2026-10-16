// rms_norm_modulate: weighted RMSNorm and an adaLN modulation in one pass.
//   out[r, d] = x[r, d] * rsqrt(mean(x[r, :]^2) + eps) * w[d] * (1 + scale[m, d]) + shift[m, d],  m = r / rowsPerMod
// IN_FP16 / OUT_FP16 pick the activation dtypes independently, so an F32 residual stream can feed an F16 block
// branch. Weight, scale and shift are F32. One workgroup per row. Matches RmsNormModulateReference.
//
// Bindings: 0=x (in), 1=weight [dim], 2=scale [mods, dim], 3=shift [mods, dim] (read only when hasShift), 4=out
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

layout(set = 0, binding = 0) readonly  buffer X_  { IN_T x[]; };
layout(set = 0, binding = 1) readonly  buffer W_  { float weight[]; };
layout(set = 0, binding = 2) readonly  buffer Sc_ { float scale[]; };
layout(set = 0, binding = 3) readonly  buffer Sh_ { float shift[]; };
layout(set = 0, binding = 4) writeonly buffer Y_  { OUT_T y[]; };

layout(push_constant) uniform Push {
    uint dim;
    uint rows;
    uint rowsPerMod;
    float eps;
    uint hasShift;
} pc;

shared float warpSums[64];
shared float gInv;

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint base = row * pc.dim;

    float sqsum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.dim; i += gl_WorkGroupSize.x) {
        float v = float(x[base + i]);
        sqsum += v * v;
    }
    sqsum = subgroupAdd(sqsum);
    if (subgroupElect()) warpSums[gl_SubgroupID] = sqsum;
    barrier();
    if (gl_SubgroupID == 0u) {
        float w = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) w += warpSums[k];
        w = subgroupAdd(w);
        if (subgroupElect()) gInv = inversesqrt(w / float(pc.dim) + pc.eps);
    }
    barrier();

    float inv = gInv;
    uint mod = (row / pc.rowsPerMod) * pc.dim;
    for (uint i = gl_LocalInvocationIndex; i < pc.dim; i += gl_WorkGroupSize.x) {
        float v = float(x[base + i]) * inv * weight[i] * (1.0 + scale[mod + i]);
        if (pc.hasShift != 0u) v += shift[mod + i];
        y[base + i] = OUT_T(v);
    }
}
