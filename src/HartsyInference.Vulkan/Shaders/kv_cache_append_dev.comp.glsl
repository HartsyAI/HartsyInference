// kv_cache_append_dev: same as kv_cache_append, but the write-slot offset comes from a DEVICE buffer
// (Pos_[1] = qOffset, the same index rope_decode_step reads) instead of a push constant — the
// decode-graph variant, since a captured/replayed command buffer cannot re-bake a per-step offset into
// its bytes. F32-only (decode-graph device state is F32-only throughout this backend).
//
// Compile:
//   glslc kv_cache_append_dev.comp.glsl -o kv_cache_append_dev_f32.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Buf_          { float buf_data[]; };
layout(set = 0, binding = 1) readonly buffer New_ { float new_data[]; };
layout(set = 0, binding = 2) readonly buffer Pos_ { uint pos_data[]; };   // [0]=kvLen, [1]=qOffset

layout(push_constant) uniform Push {
    uint heads;
    uint maxSeq;
    uint headDim;
    uint tNew;
} pc;

void main() {
    uint total = pc.heads * pc.tNew * pc.headDim;
    uint i = gl_GlobalInvocationID.x;
    if (i >= total) return;

    uint offset = pos_data[1];
    uint d = i % pc.headDim;
    uint t = (i / pc.headDim) % pc.tNew;
    uint h = i / (pc.headDim * pc.tNew);

    uint srcIdx = (h * pc.tNew + t) * pc.headDim + d;
    uint dstIdx = (h * pc.maxSeq + (offset + t)) * pc.headDim + d;
    buf_data[dstIdx] = new_data[srcIdx];
}
