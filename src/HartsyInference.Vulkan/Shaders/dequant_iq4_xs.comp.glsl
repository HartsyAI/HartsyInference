// Dequantize IQ4_XS -> F16. Mirrors native/cuda/dequant/dequant_iq4_xs_to_f16.cu exactly.
//
// IQ4_XS layout (256 elements per super-block, 136 bytes):
//   [2 bytes FP16 d] [2 bytes scales_h] [4 bytes scales_l] [128 bytes nibbles]
// Reconstruction: x = d * (ls - 32) * kvalues_iq4nl[q], ls = scales_l nibble | (scales_h bits << 4).

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define SUPER_ELEMS 256u
#define SUPER_BYTES 136u
#define SUB_ELEMS 32u

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer In_  { uint in_data[]; };
layout(set = 0, binding = 1) writeonly buffer Out_ { float16_t out_data[]; };

layout(push_constant) uniform Push { uint blockCount; } pc;

const int kvalues_iq4nl[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

uint readByte(uint byteOffset) {
    uint word = in_data[byteOffset >> 2];
    uint shift = (byteOffset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

float readHalf(uint byteOffset) {
    uint bits = readByte(byteOffset) | (readByte(byteOffset + 1u) << 8u);
    return unpackHalf2x16(bits).x;
}

void main() {
    uint total = pc.blockCount * SUPER_ELEMS;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint sb = gid / SUPER_ELEMS;
    uint tid = gid % SUPER_ELEMS;
    uint blockBase = sb * SUPER_BYTES;

    float d = readHalf(blockBase);
    uint scalesH = readByte(blockBase + 2u) | (readByte(blockBase + 3u) << 8u);
    uint ib = tid / SUB_ELEMS;
    uint i = tid % SUB_ELEMS;
    uint ls = ((readByte(blockBase + 4u + ib / 2u) >> (4u * (ib % 2u))) & 0xFu) | (((scalesH >> (2u * ib)) & 3u) << 4u);
    float dl = d * (float(int(ls)) - 32.0);
    uint b = readByte(blockBase + 8u + ib * 16u + (i & 15u));
    uint q = (i < 16u) ? (b & 0xFu) : (b >> 4u);
    out_data[gid] = float16_t(dl * float(kvalues_iq4nl[q]));
}
