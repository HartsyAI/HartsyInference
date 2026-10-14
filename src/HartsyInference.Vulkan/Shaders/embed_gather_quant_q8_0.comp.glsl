// embed_gather_quant_q8_0: decode-graph embedding gather that reads a Q8_0 table directly (no F32 copy).
// Gathers ONE row (index = the device-resident token id) and dequantizes it into a [1,1,hidden] F32 output.
// Arithmetic mirrors Codec_Q8_0.DequantizeToF32 (x = d * q) followed by the separate `* scale` the eager F32
// path applies (Granite/MiniCPM embedding scale). `precise` keeps the GPU compiler from contracting into FMA
// so the bytes-to-float result is bit-identical to the CPU dequant.
//
// Q8_0 layout: 32 elements per block, 34 bytes/block: [2 bytes FP16 scale][32 bytes int8 values].
// Push constants: hidden (uint), scale bits (uint, reinterpret as float).

#version 460

#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 34u

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
    uint blocksPerRow = pc.hidden / BLOCK_ELEMS;
    uint blockBase = (tokenId * blocksPerRow + gid / BLOCK_ELEMS) * BLOCK_BYTES;
    precise float d = readHalf(blockBase);
    precise float v = d * float(readSignedByte(blockBase + 2u + gid % BLOCK_ELEMS));
    precise float scaled = v * uintBitsToFloat(pc.scaleBits);
    out_data[gid] = scaled;
}
