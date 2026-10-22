// ltx2_split_rope: in-place LTX-2 split rotary (rotate-half inside each head) on x [S, heads*headDim], with
// cos/sin [S, heads*headDim/2] F32 (one half-width row per head). Matches IBackend.Ltx2SplitRope bit for bit:
//   a' = a*c - b*s,  b' = b*c + a*s   (a = x[i], b = x[i + headDim/2])
// `precise` keeps the driver from fusing the products into an FMA, which would round differently from the host.
//
// Built by build.sh in this directory into _f32 and _f16 variants, with the compiler that built the committed set.

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
    uint r = pc.headDim / 2;
    uint total = pc.seqLen * pc.heads * r;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint i = gid % r;
    uint t = gid / r;
    uint h = t % pc.heads;
    uint s = t / pc.heads;

    uint headBase = (s * pc.heads + h) * pc.headDim;
    uint cosBase = s * (pc.heads * r) + h * r;

    float a = TO_F32(x_data[headBase + i]);
    float b = TO_F32(x_data[headBase + i + r]);
    float c = cos_data[cosBase + i];
    float sn = sin_data[cosBase + i];

    precise float o0 = a * c - b * sn;
    precise float o1 = b * c + a * sn;
    x_data[headBase + i]     = FROM_F32(o0);
    x_data[headBase + i + r] = FROM_F32(o1);
}
