// Dequantize Q8_0 -> F16. Mirrors native/cuda/dequant/dequant_q8_0_to_f16.cu exactly.
//
// Q8_0 layout: 32 elements per block, 34 bytes/block.
//   [2 bytes FP16 scale][32 bytes int8 values]
// Reconstruction: x[i] = scale * q[i]

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 34u

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

int readSignedByte(uint byteOffset) {
    uint v = readByte(byteOffset);
    return v >= 128u ? int(v) - 256 : int(v);
}

void main() {
    uint total = pc.blockCount * BLOCK_ELEMS;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint b = gid / BLOCK_ELEMS;
    uint tid = gid % BLOCK_ELEMS;
    uint blockBase = b * BLOCK_BYTES;

    float scale = readHalf(blockBase);
    int q = readSignedByte(blockBase + 2u + tid);

    out_data[gid] = float16_t(scale * float(q));
}
