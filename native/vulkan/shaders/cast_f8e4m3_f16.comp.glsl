// cast_f8e4m3_f16: FP8 E4M3 → FP16 cast.
//
// FP8 E4M3 layout: [sign:1][exponent:4][mantissa:3], 8 bits total.
// Bias = 7 (E4M3 nominal). Subnormals when exp == 0; infinities/NaN at exp == 0xF.
//
// We unpack four FP8 values from each 32-bit word in the input buffer (since
// GLSL storage buffers can't directly index uint8 elements without
// VK_KHR_8bit_storage which we don't yet require).
//
// Bindings: 0=in (packed: ceil(N/4) uint words), 1=out (FP16, N elements)
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer In_  { uint      packed[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { float16_t outp[]; };

layout(push_constant) uniform Push {
    uint count;          // total FP8 elements
    float scale;         // optional per-tensor scale (for fp8_scaled checkpoints)
} pc;

float decode_e4m3(uint byteVal) {
    uint sign = (byteVal >> 7) & 0x1u;
    uint exp  = (byteVal >> 3) & 0xFu;
    uint mant = byteVal & 0x7u;

    // E4M3 has no infinity (exp 0xF mant 0x7 == NaN sentinel; everything else is finite).
    // For our purposes we treat NaN as 0 (model loaders shouldn't emit NaNs).
    if (exp == 0xFu && mant == 0x7u) return 0.0;
    if (exp == 0u && mant == 0u) return sign != 0u ? -0.0 : 0.0;

    float val;
    if (exp == 0u) {
        // Subnormal: value = sign * mant * 2^(1 - bias - mantissa_bits)
        //          = sign * mant * 2^-9
        val = float(mant) * 0.001953125;     // 2^-9
    } else {
        // Normal: value = sign * (1 + mant/8) * 2^(exp - 7)
        float mantissa = 1.0 + float(mant) * 0.125;     // 1 + m/8
        int e = int(exp) - 7;
        val = mantissa * exp2(float(e));
    }
    if (sign != 0u) val = -val;
    return val;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;

    uint wordIdx = i / 4u;
    uint laneIdx = i % 4u;
    uint w = packed[wordIdx];
    uint b = (w >> (laneIdx * 8u)) & 0xFFu;
    float v = decode_e4m3(b) * pc.scale;
    outp[i] = float16_t(v);
}
