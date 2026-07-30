// apply_rope: in-place split-half rotary embedding on q AND k, [B,L,H,D], sharing one cos/sin table
// indexed [B,L,D] (only the first `headDim` entries per (b,s) row are read). Matches IBackend.ApplyRope
// — q and k must have the SAME numHeads (that's the CPU reference's own contract; GQA head-count
// mismatches use ApplyRopeSingle per-tensor instead).
//
// out[i]        = in[i]*cos[i]        - in[i+half]*sin[i]
// out[i+half]   = in[i+half]*cos[i+half] + in[i]*sin[i+half]
//
// Compile:
//   glslc apply_rope.comp.glsl -o apply_rope_f32.spv
//   glslc -DUSE_FP16=1 apply_rope.comp.glsl -o apply_rope_f16.spv

#version 460

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

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Q_   { DTYPE q_data[];   };
layout(set = 0, binding = 1) buffer K_   { DTYPE k_data[];   };
layout(set = 0, binding = 2) readonly buffer Cos_ { float cos_data[]; };
layout(set = 0, binding = 3) readonly buffer Sin_ { float sin_data[]; };

layout(push_constant) uniform Push {
    uint batch;
    uint seqLen;
    uint numHeads;
    uint headDim;
} pc;

void main() {
    uint half_ = pc.headDim / 2;
    uint total = pc.batch * pc.seqLen * pc.numHeads * half_;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint i = gid % half_;
    uint t = gid / half_;
    uint h = t % pc.numHeads;
    t /= pc.numHeads;
    uint s = t % pc.seqLen;
    uint b = t / pc.seqLen;

    uint freqBase = (b * pc.seqLen + s) * pc.headDim;
    float c0 = cos_data[freqBase + i];
    float s0 = sin_data[freqBase + i];
    float c1 = cos_data[freqBase + i + half_];
    float s1 = sin_data[freqBase + i + half_];

    uint vecOff = ((b * pc.seqLen + s) * pc.numHeads + h) * pc.headDim;

    float ql = TO_F32(q_data[vecOff + i]);
    float qu = TO_F32(q_data[vecOff + i + half_]);
    q_data[vecOff + i]        = FROM_F32(ql * c0 - qu * s0);
    q_data[vecOff + i + half_] = FROM_F32(qu * c1 + ql * s1);

    float kl = TO_F32(k_data[vecOff + i]);
    float ku = TO_F32(k_data[vecOff + i + half_]);
    k_data[vecOff + i]        = FROM_F32(kl * c0 - ku * s0);
    k_data[vecOff + i + half_] = FROM_F32(ku * c1 + kl * s1);
}
