// gated_residual_last_dim: out = residual + gate*value, gate [B,D] broadcast over the seq axis of
// value/residual/output [B,seqLen,D]. Matches IBackend.GatedResidualLastDim's CPU reference exactly
// (DiT residual-gate, e.g. Flux/Krea2's attn/mlp gate). Activation (residual/value/out) may be F16 or
// F32 (USE_FP16); gate is ALWAYS F32, same convention as affine_broadcast_last_dim's scale/shift.
//
// Compile:
//   glslc gated_residual_last_dim.comp.glsl -o gated_residual_last_dim_f32.spv
//   glslc -DUSE_FP16=1 gated_residual_last_dim.comp.glsl -o gated_residual_last_dim_f16.spv

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

layout(set = 0, binding = 0) readonly  buffer Res_   { DTYPE res_data[];   };
layout(set = 0, binding = 1) readonly  buffer Val_   { DTYPE val_data[];   };
layout(set = 0, binding = 2) readonly  buffer Gate_  { float gate_data[];  };
layout(set = 0, binding = 3) writeonly buffer Out_   { DTYPE out_data[];   };

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

    out_data[i] = FROM_F32(TO_F32(res_data[i]) + gate_data[pIdx] * TO_F32(val_data[i]));
}
