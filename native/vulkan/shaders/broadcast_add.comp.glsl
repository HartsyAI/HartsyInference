// broadcast_add: hidden[B, C, ...spatial] += bias[C]   (in-place over hidden)
// Bindings: 0=hidden (in/out), 1=bias (broadcast over channel dim)
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

layout(set = 0, binding = 0) buffer Hidden_ { DTYPE hidden[]; };
layout(set = 0, binding = 1) readonly buffer Bias_ { DTYPE bias[]; };

layout(push_constant) uniform Push {
    uint channels;     // C
    uint spatial;      // product of dims past C (1 for [B, C])
    uint total;        // B * C * spatial
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint c = (i / pc.spatial) % pc.channels;
    float h = TO_F32(hidden[i]);
    float b = TO_F32(bias[c]);
    hidden[i] = FROM_F32(h + b);
}
