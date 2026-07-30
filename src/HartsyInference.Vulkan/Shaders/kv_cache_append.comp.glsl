// kv_cache_append: writes newKv [1,H,tNew,D] into buffer [1,H,maxSeq,D] at time offset `offset`, in
// place. Matches IBackend.KvCacheAppend. One thread per element of newKv (H*tNew*D total).
//
// Compile:
//   glslc kv_cache_append.comp.glsl -o kv_cache_append_f32.spv
//   glslc -DUSE_FP16=1 kv_cache_append.comp.glsl -o kv_cache_append_f16.spv

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

layout(set = 0, binding = 0) buffer Buf_          { DTYPE buf_data[]; };
layout(set = 0, binding = 1) readonly buffer New_ { DTYPE new_data[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint maxSeq;
    uint headDim;
    uint tNew;
    uint offset;
} pc;

void main() {
    uint total = pc.heads * pc.tNew * pc.headDim;
    uint i = gl_GlobalInvocationID.x;
    if (i >= total) return;

    uint d = i % pc.headDim;
    uint t = (i / pc.headDim) % pc.tNew;
    uint h = i / (pc.headDim * pc.tNew);

    uint srcIdx = (h * pc.tNew + t) * pc.headDim + d;
    uint dstIdx = (h * pc.maxSeq + (pc.offset + t)) * pc.headDim + d;
    buf_data[dstIdx] = new_data[srcIdx];
}
