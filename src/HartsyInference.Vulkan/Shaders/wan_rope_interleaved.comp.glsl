// wan_rope_interleaved: in-place GPT-J-style interleaved-pair RoPE on a single tensor
// x [S, heads, headDim] (or the equivalent flat [1,S,H,D] pre-permute layout), cos/sin
// shared across heads [S, headDim] (duplicated-pair layout: pair i stored at both 2i and
// 2i+1 — see FluxRope.GetGpuTables). Matches IBackend.WanRopeInterleaved's CPU reference
// exactly:
//   re' = re*c - im*s
//   im' = re*s + im*c
// x may be F16 or F32 (USE_FP16); cos/sin are ALWAYS F32 (small per-position tables, not
// worth halving — same convention as affine_broadcast_last_dim's scale/shift).
//
// Compile:
//   glslc wan_rope_interleaved.comp.glsl -o wan_rope_interleaved_f32.spv
//   glslc -DUSE_FP16=1 wan_rope_interleaved.comp.glsl -o wan_rope_interleaved_f16.spv

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

layout(set = 0, binding = 0) buffer X_   { DTYPE x_data[];   };
layout(set = 0, binding = 1) readonly buffer Cos_ { float cos_data[]; };
layout(set = 0, binding = 2) readonly buffer Sin_ { float sin_data[]; };

layout(push_constant) uniform Push {
    uint seqLen;
    uint heads;
    uint headDim;
} pc;

void main() {
    uint pairs = pc.headDim / 2;
    uint total = pc.seqLen * pc.heads * pairs;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint i = gid % pairs;
    uint t = gid / pairs;
    uint h = t % pc.heads;
    uint s = t / pc.heads;

    uint xoff = (s * pc.heads + h) * pc.headDim;
    uint coff = s * pc.headDim;
    uint i0 = 2u * i;

    float re = TO_F32(x_data[xoff + i0]);
    float im = TO_F32(x_data[xoff + i0 + 1u]);
    float c = cos_data[coff + i0];
    float sn = sin_data[coff + i0];

    x_data[xoff + i0]      = FROM_F32(re * c - im * sn);
    x_data[xoff + i0 + 1u] = FROM_F32(re * sn + im * c);
}
