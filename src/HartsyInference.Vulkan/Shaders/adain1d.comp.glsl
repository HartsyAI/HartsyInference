// adain1d: adaptive instance norm over time, one workgroup per (batch, channel) row.
//   out = (x - mean) * (1 + gamma) * invStd + beta,   invStd = rsqrt(var + eps)
// gamma/beta are [B, C] (perBatch = 1, index = row) or [C] (perBatch = 0, index = row % C).
// Two passes over T: mean first, then the centered sum of squares. The single-pass E[x^2] - E[x]^2
// form cancels on long non-zero-mean channels.
//
// Bindings: 0=x (in), 1=gamma, 2=beta, 3=y (out)
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_     { float x[];     };
layout(set = 0, binding = 1) readonly  buffer Gamma_ { float gamma[]; };
layout(set = 0, binding = 2) readonly  buffer Beta_  { float beta[];  };
layout(set = 0, binding = 3) writeonly buffer Y_     { float y[];     };

layout(push_constant) uniform Push {
    uint  T;
    uint  C;
    uint  perBatch;
    float eps;
} pc;

shared float warpSum[64];
shared float gMean;
shared float gInvStd;

void main() {
    uint row    = gl_WorkGroupID.x;
    uint base   = row * pc.T;
    uint affine = (pc.perBatch != 0u) ? row : (row % pc.C);

    // Phase 1: mean over T.
    float sum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.T; i += gl_WorkGroupSize.x) {
        sum += x[base + i];
    }
    sum = subgroupAdd(sum);
    if (subgroupElect()) {
        warpSum[gl_SubgroupID] = sum;
    }
    barrier();
    if (gl_SubgroupID == 0u) {
        float w = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) {
            w += warpSum[k];
        }
        w = subgroupAdd(w);
        if (subgroupElect()) {
            gMean = w / float(pc.T);
        }
    }
    barrier();

    // Phase 2: centered sum of squares, then the inverse standard deviation.
    float mean = gMean;
    float sq = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.T; i += gl_WorkGroupSize.x) {
        float d = x[base + i] - mean;
        sq += d * d;
    }
    sq = subgroupAdd(sq);
    if (subgroupElect()) {
        warpSum[gl_SubgroupID] = sq;
    }
    barrier();
    if (gl_SubgroupID == 0u) {
        float w = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) {
            w += warpSum[k];
        }
        w = subgroupAdd(w);
        if (subgroupElect()) {
            gInvStd = inversesqrt(w / float(pc.T) + pc.eps);
        }
    }
    barrier();

    // Phase 3: normalize with the adaptive affine.
    float invStd = gInvStd;
    float scaleG = 1.0 + gamma[affine];
    float shiftB = beta[affine];
    for (uint i = gl_LocalInvocationIndex; i < pc.T; i += gl_WorkGroupSize.x) {
        y[base + i] = (x[base + i] - mean) * scaleG * invStd + shiftB;
    }
}
