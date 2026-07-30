// embed_gather_decode: gathers ONE row (index = the device-resident token id) from a GPU-resident
// embedding table into a [1,1,hidden] output — the decode-graph replacement for a host-side embedding
// lookup by a CPU-known token id. Reads the token id from a device buffer (not a push constant) so the
// SAME recorded dispatch replays correctly after ArgMaxInto writes a new id between graph launches.
//
// Compile:
//   glslc embed_gather_decode.comp.glsl -o embed_gather_decode_f32.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) writeonly buffer Out_     { float out_data[]; };
layout(set = 0, binding = 1) readonly buffer Embed_    { float embed_data[]; };
layout(set = 0, binding = 2) readonly buffer TokenId_  { uint token_data[]; };

layout(push_constant) uniform Push {
    uint hidden;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.hidden) return;
    uint tokenId = token_data[0];
    out_data[gid] = embed_data[tokenId * pc.hidden + gid];
}
