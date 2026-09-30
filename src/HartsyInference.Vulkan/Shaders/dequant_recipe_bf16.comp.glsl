// dequant_recipe_bf16: DeepSeek-V4.1 checkpoint recipes to BF16 (mode 0: E2M1 nibbles with an E8M0 scale per blockCols inputs,
// mode 1: E4M3 bytes with an E8M0 scale per blockRows x blockCols block). Same decode as ModelAssets Mxfp4E8M0Codec and
// Fp8BlockE8M0Codec: value = table[q] * scale in float, then one round-to-nearest-even BF16 store. One thread per pair of outputs.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Out_ { uint out_data[]; };
layout(set = 0, binding = 1) readonly buffer Packed_ { uint packed_data[]; };
layout(set = 0, binding = 2) readonly buffer Scale_ { uint scale_data[]; };

layout(push_constant) uniform Push {
    uint mode;
    uint cols;
    uint pairs;
    uint scaleStride;
    uint scaleColOffset;
    uint blockRows;
    uint blockCols;
    uint pairBase;
} pc;

uint byteAt(bool isScale, uint i) {
    uint w = isScale ? scale_data[i >> 2] : packed_data[i >> 2];
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

// E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; nibble 8 is +0.0 as in the reference table.
float e2m1(uint nibble) {
    if (nibble == 8u) return 0.0;
    uint ex = (nibble >> 1) & 3u, man = nibble & 1u;
    float mag = ex == 0u ? 0.5 * float(man) : uintBitsToFloat((126u + ex) << 23) * (1.0 + 0.5 * float(man));
    return (nibble & 8u) != 0u ? -mag : mag;
}

float e4m3(uint b) {
    uint ex = (b >> 3) & 0xFu, man = b & 7u;
    if (ex == 15u && man == 7u) return uintBitsToFloat(0x7FC00000u);
    float mag = ex == 0u ? 0.015625 * (float(man) * 0.125) : uintBitsToFloat((120u + ex) << 23) * (1.0 + float(man) * 0.125);
    return (b & 0x80u) != 0u ? -mag : mag;
}

uint toBf16(uint bits) {
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

// Byte 255 is NaN and byte 0 is 2^-127. Devices flush float subnormals, so below scale byte 64 the product is formed 2^64
// times too large (still normal) and a result under 2^-126 is rounded straight onto the BF16 subnormal grid (steps of 2^-133).
uint element(uint e) {
    uint row = e / pc.cols, col = e - row * pc.cols;
    uint sb = byteAt(true, (row / pc.blockRows) * pc.scaleStride + pc.scaleColOffset + col / pc.blockCols);
    float q = pc.mode == 0u ? e2m1((byteAt(false, e >> 1) >> ((e & 1u) * 4u)) & 0xFu) : e4m3(byteAt(false, e));
    if (sb == 255u || q != q) return 0x7FC0u;
    if (sb >= 64u) {
        precise float v = q * uintBitsToFloat(sb << 23);
        return toBf16(floatBitsToUint(v));
    }
    precise float big = q * uintBitsToFloat((sb + 64u) << 23);
    uint bits = floatBitsToUint(big);
    if (abs(big) < uintBitsToFloat(65u << 23)) {
        return ((bits >> 16) & 0x8000u) | uint(roundEven(abs(big) * uintBitsToFloat(196u << 23)));
    }
    return toBf16(bits - (64u << 23));
}

void main() {
    uint p = pc.pairBase + gl_GlobalInvocationID.x;
    if (p >= pc.pairs) return;
    out_data[p] = element(2u * p) | (element(2u * p + 1u) << 16);
}
