// moe_gather_rows_f16: out[r, c] = f16(x[perm[r], c]) for the expert-major rows of a MoE batch, so each expert's rows are contiguous
// for its GEMMs. A negative token id (the padding MoeBuildDispatch writes past the last expert) writes zeros.
// Bindings: 0=out (f16 [rows, width]), 1=x (f32 [tokens, width]), 2=perm (i32 [rows]).
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) writeonly buffer Out_  { float16_t out_data[]; };
layout(set = 0, binding = 1) readonly  buffer X_    { float x_data[]; };
layout(set = 0, binding = 2) readonly  buffer Perm_ { int perm[]; };

layout(push_constant) uniform Push { uint rows; uint width; uint tokens; } pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = pc.rows * pc.width;
    if (gid >= total) return;
    uint r = gid / pc.width;
    uint c = gid - r * pc.width;
    int t = perm[r];
    out_data[gid] = (t < 0 || uint(t) >= pc.tokens) ? float16_t(0.0) : float16_t(x_data[uint(t) * pc.width + c]);
}
