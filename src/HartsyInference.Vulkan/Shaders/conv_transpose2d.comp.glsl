// 2D transposed convolution, F32 (Kim_Vocal_2 decoder, YOLO-seg Proto, SAM / DPT / MoGe upsampling).
// Operates on [N, C_in, H, W] -> [N, C_out, oH, oW]. Weight layout (PyTorch ConvTranspose2d, groups = 1):
// [C_in, C_out, kH, kW]. Bias: [C_out] (optional). The output size is the output tensor's, so output_padding
// needs no parameter of its own.
//
// Each invocation computes one output element by gathering: input position (iy, ix) and tap (ky, kx) contribute to
// output (oy, ox) when iy * strideH + ky - padH == oy, and the same along x. Taps that do not divide cleanly by the
// stride, or whose input position is outside the input, contribute nothing.
//
// Compile:
//   glslc conv_transpose2d.comp.glsl -o conv_transpose2d.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(constant_id = 10) const uint HAS_BIAS = 0;

layout(set = 0, binding = 0) readonly  buffer Input_  { float in_data[]; };
layout(set = 0, binding = 1) readonly  buffer Weight_ { float w_data[];  };
layout(set = 0, binding = 2) readonly  buffer Bias_   { float b_data[];  };
layout(set = 0, binding = 3) writeonly buffer Output_ { float out_data[]; };

layout(push_constant) uniform Push {
    uint batch;
    uint cIn;
    uint cOut;
    uint iH;
    uint iW;
    uint oH;
    uint oW;
    uint kH;
    uint kW;
    uint strideH;
    uint strideW;
    int  padH;
    int  padW;
} pc;

void main() {
    uint outFlat = gl_GlobalInvocationID.x;
    uint total = pc.batch * pc.cOut * pc.oH * pc.oW;
    if (outFlat >= total) return;

    uint ox = outFlat % pc.oW;
    uint oy = (outFlat / pc.oW) % pc.oH;
    uint oc = (outFlat / (pc.oW * pc.oH)) % pc.cOut;
    uint b = outFlat / (pc.oW * pc.oH * pc.cOut);

    float acc = (HAS_BIAS == 1u) ? b_data[oc] : 0.0;
    uint inPlane = pc.iH * pc.iW;
    uint wStep = pc.cOut * pc.kH * pc.kW;

    for (uint ky = 0u; ky < pc.kH; ky++) {
        int numY = int(oy) + pc.padH - int(ky);
        if (numY < 0 || uint(numY) % pc.strideH != 0u) continue;
        uint iy = uint(numY) / pc.strideH;
        if (iy >= pc.iH) continue;
        for (uint kx = 0u; kx < pc.kW; kx++) {
            int numX = int(ox) + pc.padW - int(kx);
            if (numX < 0 || uint(numX) % pc.strideW != 0u) continue;
            uint ix = uint(numX) / pc.strideW;
            if (ix >= pc.iW) continue;
            uint inBase = b * pc.cIn * inPlane + iy * pc.iW + ix;
            uint wBase = (oc * pc.kH + ky) * pc.kW + kx;
            for (uint ic = 0u; ic < pc.cIn; ic++) {
                acc += in_data[inBase + ic * inPlane] * w_data[wBase + ic * wStep];
            }
        }
    }
    out_data[outFlat] = acc;
}
