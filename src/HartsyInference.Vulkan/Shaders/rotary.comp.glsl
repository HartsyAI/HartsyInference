// rotary: generic rotary position embedding — IBackend.Rotary (docs/BACKEND_OPS.md; the oracle is RotaryReference).
// One invocation per pair over six index axes [B, H, S1, S2, S3, P]; the push constants carry each view's strides,
// the stride from a pair's first element to its second (0 on a table shares one entry), the offsets and the
// device-position row term. The arithmetic is `precise` so no contraction changes the rounding: the result matches
// the host reference bit for bit.
//
// Compile:
//   glslangValidator -V -DUSE_FP16=0 rotary.comp.glsl -o rotary_f32.spv
//   glslangValidator -V -DUSE_FP16=1 rotary.comp.glsl -o rotary_f16.spv
//
// Bindings: 0=x (in place), 1=cos, 2=sin, 3=positions (int; read only when posRowStride != 0)

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

#define AXES 6

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0)          buffer X_   { DTYPE x_data[]; };
layout(set = 0, binding = 1) readonly buffer Cos_ { float cos_data[]; };
layout(set = 0, binding = 2) readonly buffer Sin_ { float sin_data[]; };
layout(set = 0, binding = 3) readonly buffer Pos_ { int pos_data[]; };

layout(push_constant, std430) uniform Push {
    uint dims[AXES];
    uint xs[AXES];
    uint cs[AXES];
    uint ss[AXES];
    uint xSecond;
    uint cosSecond;
    uint sinSecond;
    uint xOff;
    uint cosOff;
    uint sinOff;
    uint posRowStride;
    uint posSlot;
} pc;

void main() {
    uint total = 1u;
    for (int a = 0; a < AXES; ++a) total *= pc.dims[a];
    uint linear = gl_GlobalInvocationID.x;
    if (linear >= total) return;
    uint rem = linear;
    uint xo = pc.xOff, co = pc.cosOff, so = pc.sinOff;
    for (int a = AXES - 1; a >= 0; --a) {
        uint q = rem / pc.dims[a];
        uint i = rem - q * pc.dims[a];
        xo += i * pc.xs[a];
        co += i * pc.cs[a];
        so += i * pc.ss[a];
        rem = q;
    }
    if (pc.posRowStride != 0u) {
        uint shift = uint(pos_data[pc.posSlot]) * pc.posRowStride;
        co += shift;
        so += shift;
    }
    float x0 = float(x_data[xo]), x1 = float(x_data[xo + pc.xSecond]);
    float c0 = cos_data[co], c1 = cos_data[co + pc.cosSecond];
    float s0 = sin_data[so], s1 = sin_data[so + pc.sinSecond];
    precise float r0 = x0 * c0 - x1 * s0;
    precise float r1 = x1 * c1 + x0 * s1;
    x_data[xo] = DTYPE(r0);
    x_data[xo + pc.xSecond] = DTYPE(r1);
}
