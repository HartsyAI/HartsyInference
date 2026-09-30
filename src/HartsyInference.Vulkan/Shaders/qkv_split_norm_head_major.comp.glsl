// qkv_split_norm_head_major: head-major, subset-emitting twin of qkv_split_norm.
//   qkv[token, packStride*w] -> q/k/v each [B, heads, seq, headDim]
//   q and k are RMS-normed over headDim and scaled by their per-dim weight; v is copied.
//
// packStride is how many w-wide segments the packed source holds; qSlot/kSlot/vSlot say which segment each
// output reads, negative meaning not produced. An output that is not produced still has a descriptor bound
// (Vulkan has no optional binding) and the caller binds a produced output's buffer there; every store is
// slot-guarded so nothing writes through it.
//
// A separate kernel rather than spec constants on qkv_split_norm, which is on a shipped generation path.
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

layout(set = 0, binding = 0) readonly buffer QKV_ { DTYPE qkv[]; };
layout(set = 0, binding = 1) readonly buffer QW_  { float qWeight[]; };
layout(set = 0, binding = 2) readonly buffer KW_  { float kWeight[]; };
layout(set = 0, binding = 3)          buffer Q_   { DTYPE q[]; };
layout(set = 0, binding = 4)          buffer K_   { DTYPE k[]; };
layout(set = 0, binding = 5)          buffer V_   { DTYPE v[]; };

layout(push_constant) uniform Push {
    uint headDim;     // reduction width
    uint heads;       // heads per token
    uint w;           // heads * headDim, the per-segment width
    uint tokens;      // total tokens
    uint seq;         // positions per batch item, to split a token into (b, s)
    uint packStride;  // w-wide segments in the packed source
    int  qSlot;       // segment each output reads; negative = not produced
    int  kSlot;
    int  vSlot;
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

    uint headOff = h * pc.headDim;
    uint tokBase = t * pc.packStride * pc.w;
    uint qIn = tokBase + uint(max(pc.qSlot, 0)) * pc.w + headOff;
    uint kIn = tokBase + uint(max(pc.kSlot, 0)) * pc.w + headOff;
    uint vIn = tokBase + uint(max(pc.vSlot, 0)) * pc.w + headOff;

    uint b = t / pc.seq, s = t - (t / pc.seq) * pc.seq;
    uint outOff = ((b * pc.heads + h) * pc.seq + s) * pc.headDim;

    // Sum of squares for q and k over headDim. An output that is not produced contributes nothing and its
    // inverse norm is never read, so there is no branch here — only at the loads and the stores.
    float qss = 0.0, kss = 0.0;
    for (uint d = gl_LocalInvocationIndex; d < pc.headDim; d += gl_WorkGroupSize.x) {
        float qv = pc.qSlot >= 0 ? TO_F32(qkv[qIn + d]) : 0.0;
        float kv = pc.kSlot >= 0 ? TO_F32(qkv[kIn + d]) : 0.0;
        qss += qv * qv;
        kss += kv * kv;
    }
    // The cross-subgroup fold is not optional even though headDim is small: subgroupAdd reduces within a
    // subgroup, so a workgroup spanning more than one leaves each with a partial sum, and the subgroup width
    // is hardware (32 NVIDIA, 64 AMD, as low as 8 Intel), not something the dispatch picks. Reading one
    // partial as the total normalizes by the wrong denominator and still looks plausible.
    qss = subgroupAdd(qss);
    kss = subgroupAdd(kss);
    if (subgroupElect()) { warpQ[gl_SubgroupID] = qss; warpK[gl_SubgroupID] = kss; }
    barrier();

    if (gl_SubgroupID == 0u) {
        // Strided so gl_NumSubgroups > gl_SubgroupSize is handled on small-subgroup hardware.
        float q2 = 0.0, k2 = 0.0;
        for (uint sg = gl_SubgroupInvocationID; sg < gl_NumSubgroups; sg += gl_SubgroupSize) {
            q2 += warpQ[sg];
            k2 += warpK[sg];
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
        if (pc.qSlot >= 0) q[outOff + d] = FROM_F32(TO_F32(qkv[qIn + d]) * qInv * qWeight[d]);
        if (pc.kSlot >= 0) k[outOff + d] = FROM_F32(TO_F32(qkv[kIn + d]) * kInv * kWeight[d]);
        if (pc.vSlot >= 0) v[outOff + d] = qkv[vIn + d];
    }
}
