// mask_add: scores[scoreOffset + i] += mask[maskOffset + i]   for i in [0, count)
// Used by SDPA to apply a per-(seqQ, seqKv) attention mask broadcast across heads.
// The host loops per head and dispatches once with the right scoreOffset.
//
// Bindings: 0 = scores (in/out), 1 = mask
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

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) buffer Scores_ { DTYPE scores[]; };
layout(set = 0, binding = 1) readonly buffer Mask_ { float mask[]; };

layout(push_constant) uniform Push {
    uint count;
    uint scoreOffset;
    uint maskOffset;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    float v = TO_F32(scores[pc.scoreOffset + i]);
    v += mask[pc.maskOffset + i];
    scores[pc.scoreOffset + i] = FROM_F32(v);
}
