// Dequantize Q2_K -> F16. Mirrors native/cuda/dequant/dequant_q2_k_to_f16.cu exactly.
//
// Q2_K layout (256 elements per super-block, 84 bytes):
//   [16 bytes scales: low nibble = sub-scale, high nibble = sub-min, one per 16-element run]
//   [64 bytes 2-bit quants, four per byte: byte 32h + 16·half + l holds elements at shifts 0/2/4/6]
//   [2 bytes FP16 d] [2 bytes FP16 dmin]
// Reconstruction: x = d * (sc & 0xF) * q - dmin * (sc >> 4).

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define SUPER_ELEMS 256u
#define SUPER_BYTES 84u

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

void main() {
    uint total = pc.blockCount * SUPER_ELEMS;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint sb = gid / SUPER_ELEMS;
    uint tid = gid % SUPER_ELEMS;
    uint blockBase = sb * SUPER_BYTES;

    float d = readHalf(blockBase + 80u);
    float dmin = readHalf(blockBase + 82u);

    uint h = tid / 128u;
    uint j = (tid % 128u) / 32u;
    uint half16 = (tid % 32u) / 16u;
    uint l = tid % 16u;

    uint sc = readByte(blockBase + 8u * h + 2u * j + half16);
    uint q = (readByte(blockBase + 16u + 32u * h + 16u * half16 + l) >> (2u * j)) & 3u;
    out_data[gid] = float16_t(d * float(sc & 0xFu) * float(q) - dmin * float(sc >> 4u));
}
