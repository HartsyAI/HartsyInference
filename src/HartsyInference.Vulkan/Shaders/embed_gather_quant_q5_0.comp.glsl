// embed_gather_quant_q5_0: decode-graph embedding gather that reads a Q5_0 table directly (no F32 copy).
// Gathers ONE row (index = the device-resident token id) and dequantizes it into a [1,1,hidden] F32 output.
// Arithmetic mirrors Codec_Q5_0.DequantizeToF32 (x = d * (q - 16)) followed by the separate `* scale` the eager
// F32 path applies. `precise` keeps the GPU compiler from contracting into FMA so the bytes-to-float result is
// bit-identical to the CPU dequant.
//
// Q5_0 layout: 32 elements per block, 22 bytes/block: [2 bytes FP16 scale][4 bytes high bits, uint32 LE][16 bytes nibbles].
// Element i (0..15) takes the low nibble of nibble byte i, element i+16 the high nibble of byte i; the fifth bit of
// element e is bit e of the high-bits word.
// Push constants: hidden (uint), scale bits (uint, reinterpret as float).

#version 460

#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 22u

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

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.hidden) return;
    uint tokenId = token_data[0];
    uint blocksPerRow = pc.hidden / BLOCK_ELEMS;
    uint blockBase = (tokenId * blocksPerRow + gid / BLOCK_ELEMS) * BLOCK_BYTES;
    uint tid = gid % BLOCK_ELEMS;

    precise float d = readHalf(blockBase);
    uint qh = readByte(blockBase + 2u) | (readByte(blockBase + 3u) << 8u)
            | (readByte(blockBase + 4u) << 16u) | (readByte(blockBase + 5u) << 24u);
    uint qsByte = readByte(blockBase + 6u + (tid & 15u));
    uint nib = tid < 16u ? (qsByte & 0x0Fu) : (qsByte >> 4u);
    uint bit = (qh >> tid) & 1u;
    int q = int(nib | (bit << 4u)) - 16;

    precise float v = d * float(q);
    precise float scaled = v * uintBitsToFloat(pc.scaleBits);
    out_data[gid] = scaled;
}
