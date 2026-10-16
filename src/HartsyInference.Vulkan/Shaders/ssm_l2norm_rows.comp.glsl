// ssm_l2norm_rows: per-row L2 norm then a post-scale, x = x / sqrt(sum(x^2) + eps) * postScale, in place. One
// workgroup per row (Gated-DeltaNet q/k heads). Matches GatedDeltaNetReference.
//
// Bindings: 0=x [rows, dim]
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) buffer X_ { float x[]; };

layout(push_constant) uniform Push {
    uint dim;
    uint rows;
    float eps;
    float postScale;
} pc;

shared float warpSums[64];
shared float gInv;

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint base = row * pc.dim;
    float sq = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.dim; i += gl_WorkGroupSize.x) {
        float v = x[base + i];
        sq += v * v;
    }
    sq = subgroupAdd(sq);
    if (subgroupElect()) warpSums[gl_SubgroupID] = sq;
    barrier();
    if (gl_SubgroupID == 0u) {
        float w = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) w += warpSums[k];
        w = subgroupAdd(w);
        if (subgroupElect()) gInv = 1.0 / sqrt(w + pc.eps);
    }
    barrier();
    float inv = gInv;
    for (uint i = gl_LocalInvocationIndex; i < pc.dim; i += gl_WorkGroupSize.x)
        x[base + i] = x[base + i] * inv * pc.postScale;
}
