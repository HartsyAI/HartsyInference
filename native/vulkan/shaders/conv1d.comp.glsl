// 1D convolution kernel for SEANet/codec models. Operates on channels-first
// [B, C, T]. Weight layout: [C_out, C_in/groups, K]. Bias: [C_out] (optional, gated
// via spec constant HAS_BIAS).
//
// Each invocation computes one output element (b, oc, j_out).
//
// Compile:
//   glslc conv1d.comp.glsl -o conv1d_f32.spv
//   glslc -DUSE_FP16=1 conv1d.comp.glsl -o conv1d_f16.spv

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
    uint groups;
} pc;

void main() {
    uint outFlat = gl_GlobalInvocationID.x;
    uint total = pc.batch * pc.cOut * pc.tOut;
    if (outFlat >= total) return;

    uint j = outFlat % pc.tOut;
    uint oc = (outFlat / pc.tOut) % pc.cOut;
    uint b = outFlat / (pc.tOut * pc.cOut);

    uint outPerGroup = pc.cOut / pc.groups;
    uint inPerGroup = pc.cIn / pc.groups;
    uint group = oc / outPerGroup;
    uint icStart = group * inPerGroup;

    int jSrcStart = int(j * pc.stride) - pc.padLeft;
    float acc = (HAS_BIAS == 1u) ? b_data[oc] : 0.0;

    uint wRow = oc * inPerGroup * pc.kernel;
    for (uint ic = 0u; ic < inPerGroup; ic++) {
        uint inCh = icStart + ic;
        uint inBase = (b * pc.cIn + inCh) * pc.tIn;
        uint wBase = wRow + ic * pc.kernel;
        for (uint k = 0u; k < pc.kernel; k++) {
            int src = jSrcStart + int(k * pc.dilation);
            if (src >= 0 && uint(src) < pc.tIn) {
                acc += TO_F32(in_data[inBase + uint(src)]) * TO_F32(w_data[wBase + k]);
            }
        }
    }
    out_data[outFlat] = FROM_F32(acc);
}
