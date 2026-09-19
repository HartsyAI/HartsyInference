// prelu: parametric ReLU with a PER-CHANNEL negative slope.
//   out[b, c, t] = x >= 0 ? x : alpha[c] * x
//
// Its own kernel rather than an elementwise op code because the slope is indexed by channel, not shared:
// the elementwise path carries one scalar in its push block and has no notion of the [B, C, T] layout
// this needs. A single-element alpha is the shared-slope case and is handled by the same indexing.
//
// Bindings: 0=x (in), 1=alpha (in), 2=out
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

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_ { DTYPE x[];     };
layout(set = 0, binding = 1) readonly  buffer A_ { float alpha[]; };
layout(set = 0, binding = 2) writeonly buffer O_ { DTYPE outp[];  };

layout(push_constant) uniform Push {
    uint channels;
    uint timeDim;
    uint total;
    uint perChannel;   // 0 when alpha is a single shared slope
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint c = pc.perChannel != 0u ? (i / pc.timeDim) % pc.channels : 0u;
    float v = TO_F32(x[i]);
    outp[i] = FROM_F32(v >= 0.0 ? v : alpha[c] * v);
}
