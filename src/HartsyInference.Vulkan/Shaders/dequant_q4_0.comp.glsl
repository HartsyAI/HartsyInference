// Dequantize Q4_0 -> F16. Mirrors native/cuda/dequant/dequant_q4_0_to_f16.cu exactly.
//
// Q4_0 layout: 32 elements per block, 18 bytes/block.
//   [2 bytes FP16 scale d][16 bytes qs (4-bit nibbles)]
// Reconstruction (ggml-quants.c dequantize_row_q4_0):
//   for j in 0..15:
//     y[j]    = ((qs[j] & 0x0F) - 8) * d
//     y[j+16] = ((qs[j] >>   4) - 8) * d
//
// Input is read as raw uint32 words (no 8-bit storage extension needed) via readByte's arbitrary-offset
// byte extraction — one thread per OUTPUT element (not one workgroup per quant block like the CUDA
// version), trading a little redundant per-block header re-reading for a dispatch that fits this
// codebase's existing 1D GroupCount/LocalX1D convention instead of needing a bespoke launch shape.

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 18u

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
    uint total = pc.blockCount * BLOCK_ELEMS;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint b = gid / BLOCK_ELEMS;
    uint tid = gid % BLOCK_ELEMS;
    uint blockBase = b * BLOCK_BYTES;

    float d = readHalf(blockBase);
    uint j = tid & 15u;
    uint upperHalf = tid >> 4u;
    uint qsByte = readByte(blockBase + 2u + j);
    int nib = int(upperHalf == 0u ? (qsByte & 0x0Fu) : (qsByte >> 4u)) - 8;

    out_data[gid] = float16_t(float(nib) * d);
}
