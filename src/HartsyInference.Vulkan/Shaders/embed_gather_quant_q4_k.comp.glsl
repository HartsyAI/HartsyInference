// embed_gather_quant_q4_k: decode-graph embedding gather that reads a Q4_K table directly (no F32 copy).
// Gathers ONE row (index = the device-resident token id) and dequantizes it into a [1,1,hidden] F32 output.
// Arithmetic mirrors Codec_Q4_K.DequantizeToF32 (subScale = d*sc; subMin = dmin*mm; x = subScale*q - subMin),
// then the separate `* scale` the eager F32 path applies. `precise` forbids FMA contraction so the result is
// bit-identical to the CPU dequant.
//
// Q4_K layout (256 elements per super-block, 144 bytes):
//   [2 bytes FP16 d][2 bytes FP16 dmin][12 bytes packed 6-bit scales+mins][128 bytes 4-bit quants]
// Push constants: hidden (uint), scale bits (uint, reinterpret as float).

#version 460

#define SUPER_ELEMS 256u
#define SUPER_BYTES 144u
#define SUB_ELEMS 32u

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) writeonly buffer Out_    { float out_data[]; };
layout(set = 0, binding = 1) readonly buffer Table_   { uint in_data[]; };
layout(set = 0, binding = 2) readonly buffer TokenId_ { uint token_data[]; };

layout(push_constant) uniform Push {
    uint hidden;
    uint scaleBits;
} pc;

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
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.hidden) return;
    uint tokenId = token_data[0];
    uint superPerRow = pc.hidden / SUPER_ELEMS;
    uint blockBase = (tokenId * superPerRow + gid / SUPER_ELEMS) * SUPER_BYTES;
    uint tid = gid % SUPER_ELEMS;

    precise float d = readHalf(blockBase);
    precise float dmin = readHalf(blockBase + 2u);
    uint scalesBase = blockBase + 4u;
    uint qsBase = blockBase + 16u;

    uint j = tid / SUB_ELEMS;
    uint i = tid % SUB_ELEMS;

    uint sc, mm;
    getScaleMinK4(j, scalesBase, sc, mm);
    precise float subScale = d * float(sc);
    precise float subMin = dmin * float(mm);

    uint subQuantsBase = qsBase + (j / 2u) * SUB_ELEMS;
    uint nibbleShift = (j % 2u == 0u) ? 0u : 4u;
    uint q = (readByte(subQuantsBase + i) >> nibbleShift) & 0x0Fu;

    precise float v = subScale * float(q) - subMin;
    precise float scaled = v * uintBitsToFloat(pc.scaleBits);
    out_data[gid] = scaled;
}
