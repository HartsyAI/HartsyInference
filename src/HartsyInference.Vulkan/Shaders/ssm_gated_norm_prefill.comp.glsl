// ssm_gated_norm_prefill: Qwen3.5's gated norm per (token, value head) row, in place:
// o = o * rsqrt(mean(o^2) + eps) * normW * silu(z). One workgroup per row. Matches GatedDeltaNetReference.
//
// Bindings: 0=output [rows, sv] (in place), 1=z [rows, sv], 2=normW [sv]
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) buffer O_ { float o[]; };
layout(set = 0, binding = 1) readonly buffer Z_ { float z[]; };
layout(set = 0, binding = 2) readonly buffer W_ { float normW[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint sv;
    float eps;
} pc;

shared float warpSums[64];
shared float gInv;

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint base = row * pc.sv;
    float sq = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.sv; i += gl_WorkGroupSize.x) {
        float v = o[base + i];
        sq += v * v;
    }
    sq = subgroupAdd(sq);
    if (subgroupElect()) warpSums[gl_SubgroupID] = sq;
    barrier();
    if (gl_SubgroupID == 0u) {
        float w = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) w += warpSums[k];
        w = subgroupAdd(w);
        if (subgroupElect()) gInv = 1.0 / sqrt(w / float(pc.sv) + pc.eps);
    }
    barrier();
    float inv = gInv;
    for (uint i = gl_LocalInvocationIndex; i < pc.sv; i += gl_WorkGroupSize.x) {
        float zv = z[base + i];
        o[base + i] = o[base + i] * inv * normW[i] * (zv / (1.0 + exp(-zv)));
    }
}
