// 1D transposed convolution for codec decoders (DAC / SNAC / EnCodec / Mimi).
// Operates on channels-first [B, C, T]. Weight layout (PyTorch ConvTranspose1d):
// [C_in, C_out, K]. Bias: [C_out] (optional).
//
// Each invocation computes one output element (b, oc, j_out). This uses an
// output-driven gather formulation: for output position j_out we walk all input
// positions i and kernel positions k whose target lands on (i*stride + k*dilation
// - padLeft) == j_out.
//
// Compile:
//   glslc conv_transpose1d.comp.glsl -o conv_transpose1d_f32.spv
//   glslc -DUSE_FP16=1 conv_transpose1d.comp.glsl -o conv_transpose1d_f16.spv

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
layout(constant_id = 10) const uint HAS_BIAS = 0;

layout(set = 0, binding = 0) readonly  buffer Input_  { DTYPE in_data[]; };
layout(set = 0, binding = 1) readonly  buffer Weight_ { DTYPE w_data[];  };
layout(set = 0, binding = 2) readonly  buffer Bias_   { float b_data[];  };
layout(set = 0, binding = 3) writeonly buffer Output_ { DTYPE out_data[]; };

layout(push_constant) uniform Push {
    uint batch;
    uint cIn;
    uint cOut;
    uint tIn;
    uint tOut;
    uint kernel;
    uint stride;
    int  padLeft;
    uint dilation;
} pc;

void main() {
    uint outFlat = gl_GlobalInvocationID.x;
    uint total = pc.batch * pc.cOut * pc.tOut;
    if (outFlat >= total) return;

    uint j = outFlat % pc.tOut;
    uint oc = (outFlat / pc.tOut) % pc.cOut;
    uint b = outFlat / (pc.tOut * pc.cOut);

    float acc = (HAS_BIAS == 1u) ? b_data[oc] : 0.0;

    // For target output position j, the input position i and kernel tap k must satisfy:
    //   i * stride + k * dilation - padLeft == j
    //   => i * stride = j + padLeft - k * dilation
    //   => i = (j + padLeft - k * dilation) / stride  (must divide cleanly)
    int jShifted = int(j) + pc.padLeft;
    for (uint k = 0u; k < pc.kernel; k++) {
        int num = jShifted - int(k * pc.dilation);
        if (num < 0) continue;
        if (uint(num) % pc.stride != 0u) continue;
        uint i = uint(num) / pc.stride;
        if (i >= pc.tIn) continue;
        for (uint ic = 0u; ic < pc.cIn; ic++) {
            uint inIdx = (b * pc.cIn + ic) * pc.tIn + i;
            uint wIdx = (ic * pc.cOut + oc) * pc.kernel + k;
            acc += TO_F32(in_data[inIdx]) * TO_F32(w_data[wIdx]);
        }
    }
    out_data[outFlat] = FROM_F32(acc);
}
