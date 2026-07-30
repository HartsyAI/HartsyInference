// repeat_kv_heads: GQA K/V head repeat, [B,Hkv,L,D] -> [B,Hkv*groupSize,L,D], matching HF's
// repeat_kv (each KV head's [L,D] slab is duplicated groupSize times contiguously in the
// query-head dimension). Matches IBackend.RepeatKvHeads's CPU reference exactly. F16 or F32
// (USE_FP16), same dtype in and out (pure gather, no arithmetic).
//
// Compile:
//   glslc repeat_kv_heads.comp.glsl -o repeat_kv_heads_f32.spv
//   glslc -DUSE_FP16=1 repeat_kv_heads.comp.glsl -o repeat_kv_heads_f16.spv

#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#else
#define DTYPE float
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly  buffer In_  { DTYPE in_data[];  };
layout(set = 0, binding = 1) writeonly buffer Out_ { DTYPE out_data[]; };

layout(push_constant) uniform Push {
    uint kvHeads;
    uint groupSize;
    uint seqLen;
    uint headDim;
    uint total;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;

    uint perHead = pc.seqLen * pc.headDim;
    uint qHeads = pc.kvHeads * pc.groupSize;

    uint d = i % perHead;
    uint t = i / perHead;
    uint qHead = t % qHeads;
    uint b = t / qHeads;

    uint kvHead = qHead / pc.groupSize;
    uint srcOff = (b * pc.kvHeads + kvHead) * perHead + d;

    out_data[i] = in_data[srcOff];
}
