// Dequantize Q3_K -> F16. Mirrors native/cuda/dequant/dequant_q3_k_to_f16.cu exactly.
//
// Q3_K layout (256 elements per super-block, 110 bytes):
//   [32 bytes hmask: bit (4h + j) of byte 16·half + l is the INVERTED high bit]
//   [64 bytes 2-bit low quants, four per byte, laid out as Q2_K's]
//   [12 bytes packed 6-bit signed scales, one per 16-element run]
//   [2 bytes FP16 d]
// Reconstruction: x = d * scale * (q - (mask bit set ? 0 : 4)).

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define SUPER_ELEMS 256u
#define SUPER_BYTES 110u

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer In_  { uint in_data[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { float16_t out_data[]; };

layout(push_constant) uniform Push { uint blockCount; } pc;

uint readByte(uint byteOffset) {
    uint word = in_data[byteOffset >> 2];
    uint shift = (byteOffset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

float readHalf(uint byteOffset) {
    uint bits = readByte(byteOffset) | (readByte(byteOffset + 1u) << 8u);
    return unpackHalf2x16(bits).x;
}

// Canonical ggml 6-bit signed scale unpack (dequantize_row_q3_K): entry s takes its low nibble from byte s (s < 8)
// or the high nibble of byte s - 8, and its two high bits from byte 8 + s % 4 at bit 2*(s / 4).
int unpackScale(uint scalesBase, uint index) {
    uint low = (index < 8u) ? (readByte(scalesBase + index) & 0xFu) : (readByte(scalesBase + index - 8u) >> 4u);
    uint high = (readByte(scalesBase + 8u + (index & 3u)) >> (2u * (index >> 2u))) & 3u;
    return int(low | (high << 4u)) - 32;
}

void main() {
    uint total = pc.blockCount * SUPER_ELEMS;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint sb = gid / SUPER_ELEMS;
    uint tid = gid % SUPER_ELEMS;
    uint blockBase = sb * SUPER_BYTES;

    float d = readHalf(blockBase + 108u);

    uint h = tid / 128u;
    uint j = (tid % 128u) / 32u;
    uint half16 = (tid % 32u) / 16u;
    uint l = tid % 16u;

    float scale = d * float(unpackScale(blockBase + 96u, 8u * h + 2u * j + half16));
    int q = int((readByte(blockBase + 32u + 32u * h + 16u * half16 + l) >> (2u * j)) & 3u);
    uint maskBit = 1u << (4u * h + j);
    int hbit = ((readByte(blockBase + 16u * half16 + l) & maskBit) != 0u) ? 0 : 4;
    out_data[gid] = float16_t(scale * float(q - hbit));
}
