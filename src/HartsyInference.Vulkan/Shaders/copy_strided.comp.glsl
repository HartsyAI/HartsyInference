// copy_strided: out[dst(i)] = in[src(i)] over up to six axes — IBackend.CopyStrided (docs/BACKEND_OPS.md; the
// oracle is StridedCopyReference). A pure move of WIDTH-byte words; the output is bound read-write because a
// scatter into a slot of a larger tensor leaves the rest of it untouched. The host contract has already collapsed
// the axes, so most dispatches carry one to three.
//
// Compile:
//   glslangValidator -V -DWIDTH=2 copy_strided.comp.glsl -o copy_strided_u16.spv
//   glslangValidator -V -DWIDTH=4 copy_strided.comp.glsl -o copy_strided_u32.spv
//   glslangValidator -V -DWIDTH=8 copy_strided.comp.glsl -o copy_strided_u64.spv

#version 460

#ifndef WIDTH
#define WIDTH 4
#endif

#if WIDTH == 2
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#define WORD uint16_t
#elif WIDTH == 8
#define WORD uvec2
#else
#define WORD uint
#endif

#define MAX_RANK 6

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly buffer In_  { WORD in_data[];  };
layout(set = 0, binding = 1)          buffer Out_ { WORD out_data[]; };

layout(push_constant, std430) uniform Push {
    uint total;
    uint rank;
    uint dims[MAX_RANK];
    uint src[MAX_RANK];
    uint dst[MAX_RANK];
    uint srcOff;
    uint dstOff;
} pc;

void main() {
    uint linear = gl_GlobalInvocationID.x;
    if (linear >= pc.total) return;
    uint rem = linear;
    uint s = pc.srcOff;
    uint d = pc.dstOff;
    for (int a = MAX_RANK - 1; a >= 0; --a) {
        if (a < int(pc.rank)) {
            uint q = rem / pc.dims[a];
            uint i = rem - q * pc.dims[a];
            s += i * pc.src[a];
            d += i * pc.dst[a];
            rem = q;
        }
    }
    out_data[d] = in_data[s];
}
