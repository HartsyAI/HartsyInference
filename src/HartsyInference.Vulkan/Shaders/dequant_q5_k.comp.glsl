// Dequantize Q5_K -> F16. Mirrors native/cuda/dequant/dequant_q5_k_to_f16.cu exactly.
//
// Q5_K layout (256 elements per super-block, 176 bytes):
//   [2 bytes FP16 d][2 bytes FP16 dmin]
//   [12 bytes packed 6-bit scales+mins]
//   [32 bytes high bits (1 per element)]
//   [128 bytes low nibbles (2 per byte, same packing as Q4_K)]
// Reconstruction: x = d * sc_j * q - dmin * m_j, q = low | (high << 4) in [0..31].

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define SUPER_ELEMS 256u
#define SUPER_BYTES 176u
#define SUB_ELEMS 32u

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

void getScaleMinK4(uint j, uint scalesBase, out uint sc, out uint mm) {
    if (j < 4u) {
        sc = readByte(scalesBase + j) & 63u;
        mm = readByte(scalesBase + j + 4u) & 63u;
    } else {
        sc = (readByte(scalesBase + j + 4u) & 0x0Fu) | ((readByte(scalesBase + j - 4u) >> 6u) << 4u);
        mm = (readByte(scalesBase + j + 4u) >> 4u) | ((readByte(scalesBase + j) >> 6u) << 4u);
    }
}

void main() {
    uint total = pc.blockCount * SUPER_ELEMS;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint sb = gid / SUPER_ELEMS;
    uint tid = gid % SUPER_ELEMS;
    uint blockBase = sb * SUPER_BYTES;

    float d = readHalf(blockBase);
    float dmin = readHalf(blockBase + 2u);
    uint scalesBase = blockBase + 4u;
    uint highBase = blockBase + 16u;
    uint lowBase = blockBase + 48u;

    uint j = tid / SUB_ELEMS;
    uint i = tid % SUB_ELEMS;

    uint sc, mm;
    getScaleMinK4(j, scalesBase, sc, mm);

    uint subLowBase = lowBase + (j / 2u) * SUB_ELEMS;
    uint lowShift = (j % 2u == 0u) ? 0u : 4u;
    uint low = (readByte(subLowBase + i) >> lowShift) & 0x0Fu;
    uint high = ((readByte(highBase + i) >> j) & 0x01u) << 4u;
    uint q = low | high;

    out_data[gid] = float16_t(d * float(sc) * float(q) - dmin * float(mm));
}
