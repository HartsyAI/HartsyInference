// qkv_split_norm: fused QKV split + per-head QK-RMSNorm.
//   qkv[t, 3w] -> q[t, w], k[t, w], v[t, w]
//   q and k are RMS-normed over headDim and scaled by their per-dim weight; v is copied.
//
// One workgroup per (token, head), reducing over headDim.
//
// The cross-subgroup fold is not optional even though headDim is small. subgroupAdd reduces within a
// subgroup, so a workgroup that spans more than one leaves each with a partial sum — and the subgroup
// width is hardware, not something the dispatch picks: 32 on NVIDIA, 64 on AMD, as low as 8 on Intel.
// Reading one subgroup's partial as the total would normalize by the wrong denominator and still look
// plausible, which is the failure mode worth spending shared memory to avoid.
//
// Bindings: 0=qkv (in), 1=qWeight (in, per-dim), 2=kWeight (in, per-dim), 3=q (out), 4=k (out), 5=v (out)
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

layout(set = 0, binding = 0) readonly  buffer QKV_ { DTYPE qkv[]; };
layout(set = 0, binding = 1) readonly  buffer QW_  { float qWeight[]; };
layout(set = 0, binding = 2) readonly  buffer KW_  { float kWeight[]; };
layout(set = 0, binding = 3) writeonly buffer Q_   { DTYPE q[]; };
layout(set = 0, binding = 4) writeonly buffer K_   { DTYPE k[]; };
layout(set = 0, binding = 5) writeonly buffer V_   { DTYPE v[]; };

layout(push_constant) uniform Push {
    uint headDim;   // reduction width
    uint heads;     // heads per token
    uint w;         // heads * headDim, the per-segment width
    uint tokens;    // total tokens
    float eps;
} pc;

shared float warpQ[64];
shared float warpK[64];
shared float gQInv;
shared float gKInv;

void main() {
    uint pair = gl_WorkGroupID.x;               // token * heads + head
    if (pair >= pc.tokens * pc.heads) return;
    uint t = pair / pc.heads;
    uint h = pair - t * pc.heads;

    uint baseIn = t * 3u * pc.w + h * pc.headDim;   // q segment for this head
    uint kIn    = baseIn + pc.w;
    uint vIn    = baseIn + 2u * pc.w;
    uint outOff = t * pc.w + h * pc.headDim;

    // Sum of squares for q and k over headDim.
    float qss = 0.0, kss = 0.0;
    for (uint d = gl_LocalInvocationIndex; d < pc.headDim; d += gl_WorkGroupSize.x) {
        float qv = TO_F32(qkv[baseIn + d]);
        float kv = TO_F32(qkv[kIn + d]);
        qss += qv * qv;
        kss += kv * kv;
    }
    qss = subgroupAdd(qss);
    kss = subgroupAdd(kss);
    if (subgroupElect()) { warpQ[gl_SubgroupID] = qss; warpK[gl_SubgroupID] = kss; }
    barrier();

    if (gl_SubgroupID == 0u) {
        // Strided so gl_NumSubgroups > gl_SubgroupSize is handled on small-subgroup hardware.
        float q2 = 0.0, k2 = 0.0;
        for (uint s = gl_SubgroupInvocationID; s < gl_NumSubgroups; s += gl_SubgroupSize) {
            q2 += warpQ[s];
            k2 += warpK[s];
        }
        q2 = subgroupAdd(q2);
        k2 = subgroupAdd(k2);
        if (subgroupElect()) {
            float invN = 1.0 / float(pc.headDim);
            gQInv = inversesqrt(q2 * invN + pc.eps);
            gKInv = inversesqrt(k2 * invN + pc.eps);
        }
    }
    barrier();

    float qInv = gQInv, kInv = gKInv;
    for (uint d = gl_LocalInvocationIndex; d < pc.headDim; d += gl_WorkGroupSize.x) {
        q[outOff + d] = FROM_F32(TO_F32(qkv[baseIn + d]) * qInv * qWeight[d]);
        k[outOff + d] = FROM_F32(TO_F32(qkv[kIn    + d]) * kInv * kWeight[d]);
        v[outOff + d] = qkv[vIn + d];
    }
}
