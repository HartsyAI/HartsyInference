// pad_strided: F.pad over up to eight axes — IBackend.Pad (docs/BACKEND_OPS.md; the oracle is StridedCopyReference).
// One invocation per OUTPUT word: each axis maps the output position back into the input, clamped (replicate),
// mirrored (reflect) or marked zero. WIDTH-byte words, no arithmetic.
//
// Compile:
//   glslangValidator -V -DWIDTH=2 pad_strided.comp.glsl -o pad_strided_u16.spv
//   glslangValidator -V -DWIDTH=4 pad_strided.comp.glsl -o pad_strided_u32.spv
//   glslangValidator -V -DWIDTH=8 pad_strided.comp.glsl -o pad_strided_u64.spv

#version 460

#ifndef WIDTH
#define WIDTH 4
#endif

#if WIDTH == 2
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#define WORD uint16_t
#define ZERO uint16_t(0)
#elif WIDTH == 8
#define WORD uvec2
#define ZERO uvec2(0, 0)
#else
#define WORD uint
#define ZERO 0u
#endif

#define MAX_RANK 8
#define MODE_ZERO 0
#define MODE_REPLICATE 1
#define MODE_REFLECT 2

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly  buffer In_  { WORD in_data[];  };
layout(set = 0, binding = 1) writeonly buffer Out_ { WORD out_data[]; };

layout(push_constant, std430) uniform Push {
    uint total;
    uint rank;
    uint mode;
    uint outDims[MAX_RANK];
    uint inDims[MAX_RANK];
    uint before[MAX_RANK];
} pc;

void main() {
    uint linear = gl_GlobalInvocationID.x;
    if (linear >= pc.total) return;
    uint rem = linear;
    uint src = 0u;
    uint inStride = 1u;
    bool zero = false;
    for (int a = MAX_RANK - 1; a >= 0; --a) {
        if (a < int(pc.rank)) {
            uint q = rem / pc.outDims[a];
            int o = int(rem - q * pc.outDims[a]);
            rem = q;
            int n = int(pc.inDims[a]);
            int i = o - int(pc.before[a]);
            if (i < 0 || i >= n) {
                if (pc.mode == MODE_REPLICATE) i = i < 0 ? 0 : n - 1;
                else if (pc.mode == MODE_REFLECT) i = i < 0 ? -i : 2 * n - 2 - i;
                else zero = true;
            }
            src += uint(i) * inStride;
            inStride *= uint(n);
        }
    }
    out_data[linear] = zero ? ZERO : in_data[src];
}
