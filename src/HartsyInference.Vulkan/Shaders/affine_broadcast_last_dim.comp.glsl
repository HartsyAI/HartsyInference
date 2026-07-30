// affine_broadcast_last_dim: out = in*scale + (shift ?? 0), broadcasting scale/shift [B,C] (or [C], the
// same [B*C] flat buffer with B effectively 1) over input/output [B,seqLen,C]. Matches
// IBackend.AffineBroadcastLastDim's CPU reference exactly (DiT adaLN modulation — Flux/Krea2/Ideogram4).
// Activation (in/out) may be F16 or F32 (USE_FP16); scale/shift are ALWAYS F32, matching CudaBackend's
// "F32 scale/shift" convention (they're small per-block modulation vectors, not worth halving).
// HAS_SHIFT=0 is Ideogram 4's scale-only adaLN (shift is null).
//
// Compile:
//   glslc -DUSE_FP16=0 -DHAS_SHIFT=1 affine_broadcast_last_dim.comp.glsl -o affine_broadcast_last_dim_f32.spv
//   glslc -DUSE_FP16=1 -DHAS_SHIFT=1 affine_broadcast_last_dim.comp.glsl -o affine_broadcast_last_dim_f16.spv
//   glslc -DUSE_FP16=0 -DHAS_SHIFT=0 affine_broadcast_last_dim.comp.glsl -o affine_broadcast_last_dim_noshift_f32.spv
//   glslc -DUSE_FP16=1 -DHAS_SHIFT=0 affine_broadcast_last_dim.comp.glsl -o affine_broadcast_last_dim_noshift_f16.spv

#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif
#ifndef HAS_SHIFT
#define HAS_SHIFT 1
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

layout(set = 0, binding = 0) readonly  buffer In_    { DTYPE in_data[]; };
layout(set = 0, binding = 1) readonly  buffer Scale_ { float scale_data[]; };
#if HAS_SHIFT == 1
layout(set = 0, binding = 2) readonly  buffer Shift_ { float shift_data[]; };
layout(set = 0, binding = 3) writeonly buffer Out_   { DTYPE out_data[]; };
#else
layout(set = 0, binding = 2) writeonly buffer Out_   { DTYPE out_data[]; };
#endif

layout(push_constant) uniform Push {
    uint dim;
    uint seqLen;
    uint total;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;

    uint d = i % pc.dim;
    uint row = i / pc.dim;
    uint b = row / pc.seqLen;
    uint pIdx = b * pc.dim + d;

    float v = TO_F32(in_data[i]) * scale_data[pIdx];
#if HAS_SHIFT == 1
    v += shift_data[pIdx];
#endif
    out_data[i] = FROM_F32(v);
}
