// situ_glu: MOSS-Image's SiTU-GLU over the [u | g] halves of the last dimension.
//   out[r, j] = (4*tanh(g/4)*sigmoid(g)) * (25*tanh(u/25)),  u = in[r, j], g = in[r, inner + j]
// Matches SituGluReference (sign-aware sigmoid, so neither tail overflows). One invocation per output element.
//
// Bindings: 0=in [rows, 2*inner], 1=out [rows, inner]
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

layout(set = 0, binding = 0) readonly  buffer In_  { DTYPE inp[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { DTYPE outp[]; };

layout(push_constant) uniform Push {
    uint inner;
    uint total;      // rows * inner
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint row = i / pc.inner;
    uint j = i - row * pc.inner;
    uint at = row * 2u * pc.inner + j;
    float u = TO_F32(inp[at]);
    float g = TO_F32(inp[at + pc.inner]);
    float sig = g >= 0.0 ? 1.0 / (1.0 + exp(-g)) : exp(g) / (1.0 + exp(g));
    outp[i] = FROM_F32(4.0 * tanh(g * 0.25) * sig * (25.0 * tanh(u / 25.0)));
}
