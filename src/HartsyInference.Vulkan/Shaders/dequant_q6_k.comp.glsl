// Dequantize Q6_K -> F16. Mirrors native/cuda/dequant/dequant_q6_k_to_f16.cu exactly.
//
// Q6_K layout (256 elements per super-block, 210 bytes):
//   [128 bytes ql (low 4 bits per element)]
//   [64 bytes qh (high 2 bits per element, packed 4-per-byte)]
//   [16 bytes int8 scales (one per 16-element sub-block)]
//   [2 bytes FP16 d (super-block scale)]
//
// Reconstruction (canonical ggml dequantize_row_q6_K): 256 elements processed in 2 halves of 128.
// Each half consumes 64 bytes ql, 32 bytes qh, 8 scales. Inner pattern over l in [0..31] writes 4
// elements at strides {+0,+32,+64,+96} with sub-block scale index scH[isOffset+{0,2,4,6}] where
// isOffset = l/16 (alternating between two scale rows per 16-element half).
//
// Dispatch: one thread per (superblock, tid in [0,64)) — each thread emits 4 outputs, unlike the other
// 5 dequant shaders (one thread per output element) — this format's algorithm is inherently 4-wide per
// thread, matching the CUDA reference's own launch shape (blockDim.x = 64) rather than forcing a
// mismatched one-element-per-thread reformulation.

#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define SUPER_ELEMS 256u
#define SUPER_BYTES 210u

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
    uint total = pc.blockCount * 64u;
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= total) return;

    uint sb = gid / 64u;
    uint tid = gid % 64u;
    uint blockBase = sb * SUPER_BYTES;

    uint qlBase = blockBase;
    uint qhBase = blockBase + 128u;
    uint scalesBase = blockBase + 192u;
    float d = readHalf(blockBase + 208u);

    uint half_ = tid / 32u;
    uint l = tid % 32u;
    uint isOffset = l / 16u;
    uint qlH = qlBase + half_ * 64u;
    uint qhH = qhBase + half_ * 32u;
    uint scH = scalesBase + half_ * 8u;
    uint halfBaseElem = half_ * 128u;

    uint qlH_l = readByte(qlH + l);
    uint qlH_l32 = readByte(qlH + l + 32u);
    uint qhH_l = readByte(qhH + l);

    int q1 = int((qlH_l & 0x0Fu) | (((qhH_l >> 0u) & 0x03u) << 4u)) - 32;
    int q2 = int((qlH_l32 & 0x0Fu) | (((qhH_l >> 2u) & 0x03u) << 4u)) - 32;
    int q3 = int((qlH_l >> 4u) | (((qhH_l >> 4u) & 0x03u) << 4u)) - 32;
    int q4 = int((qlH_l32 >> 4u) | (((qhH_l >> 6u) & 0x03u) << 4u)) - 32;

    int sc0 = readSignedByte(scH + isOffset + 0u);
    int sc2 = readSignedByte(scH + isOffset + 2u);
    int sc4 = readSignedByte(scH + isOffset + 4u);
    int sc6 = readSignedByte(scH + isOffset + 6u);

    uint base = sb * SUPER_ELEMS + halfBaseElem;
    out_data[base + l]        = float16_t(d * float(sc0) * float(q1));
    out_data[base + l + 32u]  = float16_t(d * float(sc2) * float(q2));
    out_data[base + l + 64u]  = float16_t(d * float(sc4) * float(q3));
    out_data[base + l + 96u]  = float16_t(d * float(sc6) * float(q4));
}
