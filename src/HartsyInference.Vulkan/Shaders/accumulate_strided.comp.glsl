// accumulate_strided: out[dst(i)] += in[src(i)] over up to eight axes, in place — IBackend.AccumulateStrided
// (docs/BACKEND_OPS.md; the oracle is StridedCopyReference). The add is F32 and rounds to the storage type; the
// destination addresses every element once (contract), so no atomics and no order dependence.
//
// Compile:
//   glslangValidator -V -DUSE_FP16=0 accumulate_strided.comp.glsl -o accumulate_strided_f32.spv
//   glslangValidator -V -DUSE_FP16=1 accumulate_strided.comp.glsl -o accumulate_strided_f16.spv

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

#define MAX_RANK 8

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly buffer In_  { DTYPE in_data[];  };
layout(set = 0, binding = 1)          buffer Out_ { DTYPE out_data[]; };

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
    out_data[d] = DTYPE(float(out_data[d]) + float(in_data[s]));
}
