// slice_rows: out[i] = in[elemOffset + i] — contiguous row-block slice (elemOffset = rowOffset*dim of
// the last axis). Matches IBackend.SliceRows exactly. Used by Krea2Transformer.SliceTail to split the
// joint img+txt sequence back into the image tail after the last DiT block.
//
// Compile:
//   glslc slice_rows.comp.glsl -o slice_rows_f32.spv
//   glslc -DUSE_FP16=1 slice_rows.comp.glsl -o slice_rows_f16.spv

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
    uint elemOffset;
    uint total;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    out_data[i] = in_data[pc.elemOffset + i];
}
