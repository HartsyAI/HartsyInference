// embed_gather_quant_q6_k: decode-graph embedding gather that reads a Q6_K table directly (no F32 copy).
// Gathers ONE row (index = the device-resident token id) and dequantizes it into a [1,1,hidden] F32 output.
// Arithmetic mirrors Codec_Q6_K.DequantizeToF32 (x = (d * sc) * q, with q = 6-bit value - 32) followed by the
// separate `* scale` the eager F32 path applies. The two products are separate `precise` statements in the CPU's
// left-to-right order, so the result is bit-identical to the CPU dequant.
//
// Q6_K layout (256 elements per super-block, 210 bytes):
//   [128 bytes ql (low 4 bits per element)][64 bytes qh (high 2 bits, 4 per byte)]
//   [16 bytes int8 scales (one per 16-element sub-block)][2 bytes FP16 d]
// Element i of a super-block: half = i/128, r = i%128, group = r/32, l = r%32. Its low four bits are the low nibble
// of ql[half*64 + (group&1)*32 + l] for groups 0 and 1, the high nibble for groups 2 and 3; its high two bits are
// qh[half*32 + l] >> (2*group). Its sub-block scale is scales[half*8 + l/16 + 2*group].
// Push constants: hidden (uint), scale bits (uint, reinterpret as float).

#version 460

#define SUPER_ELEMS 256u
#define SUPER_BYTES 210u

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

int readSignedByte(uint byteOffset) {
    uint v = readByte(byteOffset);
    return v >= 128u ? int(v) - 256 : int(v);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.hidden) return;
    uint tokenId = token_data[0];
    uint superPerRow = pc.hidden / SUPER_ELEMS;
    uint blockBase = (tokenId * superPerRow + gid / SUPER_ELEMS) * SUPER_BYTES;
    uint i = gid % SUPER_ELEMS;

    uint half_ = i / 128u;
    uint r = i % 128u;
    uint group = r / 32u;
    uint l = r % 32u;

    uint qlByte = readByte(blockBase + half_ * 64u + (group & 1u) * 32u + l);
    uint lowNibble = group < 2u ? (qlByte & 0x0Fu) : (qlByte >> 4u);
    uint qhByte = readByte(blockBase + 128u + half_ * 32u + l);
    uint highTwo = (qhByte >> (2u * group)) & 0x03u;
    int q = int(lowNibble | (highTwo << 4u)) - 32;

    int sc = readSignedByte(blockBase + 192u + half_ * 8u + (l / 16u) + 2u * group);
    precise float d = readHalf(blockBase + 208u);

    precise float subScale = d * float(sc);
    precise float v = subScale * float(q);
    precise float scaled = v * uintBitsToFloat(pc.scaleBits);
    out_data[gid] = scaled;
}
