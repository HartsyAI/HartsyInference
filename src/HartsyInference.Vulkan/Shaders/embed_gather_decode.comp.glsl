// embed_gather_decode: gathers ONE row (index = the device-resident token id) from a GPU-resident
// embedding table into a [1,1,hidden] output — the decode-graph replacement for a host-side embedding
// lookup by a CPU-known token id. Reads the token id from a device buffer (not a push constant) so the
// SAME recorded dispatch replays correctly after ArgMaxInto writes a new id between graph launches.
//
// TABLE_DTYPE selects the table's storage format. The output is always F32.
//   0 = F32 table                 (embed_gather_decode.spv)
//   1 = F16 table, IEEE binary16  (embed_gather_decode_f16.spv), widened by unpackHalf2x16
//   2 = BF16 table                (embed_gather_decode_bf16.spv), the upper 16 bits of an F32
// A narrow table is bound as 32-bit words, two elements per word with element 2k in the low half, so no 16-bit
// storage extension is needed. The host refuses a narrow table with an odd element count, because the last word
// would then run past the end of the buffer.
//
// Compile (Shaders/build.sh emits all three variants):
//   glslc -DTABLE_DTYPE=1 embed_gather_decode.comp.glsl -o embed_gather_decode_f16.spv

#version 460

#ifndef TABLE_DTYPE
#define TABLE_DTYPE 0
#endif

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) writeonly buffer Out_     { float out_data[]; };
#if TABLE_DTYPE == 0
layout(set = 0, binding = 1) readonly buffer Embed_    { float embed_data[]; };
#else
layout(set = 0, binding = 1) readonly buffer Embed_    { uint embed_words[]; };
#endif
layout(set = 0, binding = 2) readonly buffer TokenId_  { uint token_data[]; };

layout(push_constant) uniform Push {
    uint hidden;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.hidden) return;
    uint tokenId = token_data[0];
#if TABLE_DTYPE == 0
    out_data[gid] = embed_data[tokenId * pc.hidden + gid];
#else
    uint elem = tokenId * pc.hidden + gid;
    uint bits = (embed_words[elem >> 1u] >> ((elem & 1u) * 16u)) & 0xFFFFu;
#if TABLE_DTYPE == 1
    out_data[gid] = unpackHalf2x16(bits).x;
#else
    out_data[gid] = uintBitsToFloat(bits << 16u);
#endif
#endif
}
