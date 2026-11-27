// 3D block-sparse attention, stage 1 of 3 (block means). BlockSparseAttentionReference in Core defines it;
// block_sparse_attention.cu is the CUDA form. Operands are head-major [B, H, S, D], F32 or BF16 (OPERAND_BF16); a BF16
// value is widened to F32, exactly, as it is loaded.
//
// Per (block, head, batch): the mean row of one block of queries or of keys, [B, H, blocks, D], F32.
//
// This stage decides which blocks are read, and that is a comparison. Every value is declared `precise`, so the compiler
// may not contract or reorder, and the rows are summed in the block's raster order, the reference's. The adds are then the
// reference's bit for bit, provided the driver keeps subnormal values: no float-controls mode is declared here, and
// without one a driver may flush them to zero. The Vulkan specification does not require a divide to be correctly rounded,
// and on an RTX 3060 it is not always, so the divide goes through divRn (div_rn.glsl), which is.
#version 460
#extension GL_GOOGLE_include_directive : require
#if OPERAND_BF16
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#endif

layout(local_size_x = 128) in;

#if OPERAND_BF16
layout(set = 0, binding = 0) readonly  buffer X_ { uint16_t x_data[]; };
#else
layout(set = 0, binding = 0) readonly  buffer X_ { float x_data[]; };
#endif
layout(set = 0, binding = 1) writeonly buffer M_ { float means[]; };

layout(push_constant) uniform Push {
    uint tokens;
    uint dim;
    uint heads;
    uint blocks;
    uint gridH;
    uint gridW;
    uint bt;
    uint bh;
    uint bw;
} pc;

#include "block_sparse_common.glsl"
#include "div_rn.glsl"

float load(uint at) {
#if OPERAND_BF16
    return uintBitsToFloat(uint(x_data[at]) << 16);
#else
    return x_data[at];
#endif
}

void main() {
    uint n = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint head = b * pc.heads + h;
    uint rows = head * pc.tokens * pc.dim;
    uint len = pc.bt * pc.bh * pc.bw;
    for (uint d = gl_LocalInvocationID.x; d < pc.dim; d += 128u) {
        precise float sum = 0.0;
        for (uint o = 0u; o < len; ++o)
            sum = sum + load(rows + bsaRow(n, o, pc.gridH, pc.gridW, pc.bt, pc.bh, pc.bw) * pc.dim + d);
        precise float mean = divRn(sum, float(len));
        means[(head * pc.blocks + n) * pc.dim + d] = mean;
    }
}
