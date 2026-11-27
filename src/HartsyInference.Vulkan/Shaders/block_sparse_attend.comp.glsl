// 3D block-sparse attention, stage 3 of 3 (attention). BlockSparseAttentionReference in Core defines it;
// block_sparse_attention.cu is the CUDA form. Operands are head-major [B, H, S, D], F32 or BF16 (OPERAND_BF16); the output
// is F32 or BF16 rounded to nearest even (OUTPUT_BF16). All arithmetic is F32 and is compared by tolerance.
//
// Per (query tile, head, batch): each of up to QT queries of one query block attends the keys of the block's selected key
// blocks, one softmax kept online (running maximum and sum) over chunks of KT key rows. The 128 invocations take, per
// chunk: the row numbers, K; the scores as 2 x 2 query-key tiles; one query's softmax each; V into K's place; and 2 query
// rows by 16 channels each of the weighted sum, which stays in registers. With no selected block there is no chunk and the
// output is zero.
#version 460
#extension GL_GOOGLE_include_directive : require
#if OPERAND_BF16 || OUTPUT_BF16
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#endif

#define MAX_DIM 128
#define QT 32u
#define KT 16u

layout(local_size_x = 128) in;

#if OPERAND_BF16
layout(set = 0, binding = 0) readonly  buffer Q_ { uint16_t q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { uint16_t k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { uint16_t v_data[]; };
#else
layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { float k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { float v_data[]; };
#endif
layout(set = 0, binding = 3) readonly  buffer S_ { int selected[]; };
#if OUTPUT_BF16
layout(set = 0, binding = 4) writeonly buffer O_ { uint16_t o_data[]; };
#else
layout(set = 0, binding = 4) writeonly buffer O_ { float o_data[]; };
#endif

layout(push_constant) uniform Push {
    uint queryTokens;
    uint keyTokens;
    uint dim;
    uint heads;
    uint queryBlocks;
    uint chosen;
    uint qGridH;
    uint qGridW;
    uint qbt;
    uint qbh;
    uint qbw;
    uint kGridH;
    uint kGridW;
    uint kbt;
    uint kbh;
    uint kbw;
    float scale;
    uint firstGroup;
} pc;

// About 27 KB, under the 32 KiB the backend requires of a device.
shared float s_q[QT * MAX_DIM];
shared float s_kv[KT * MAX_DIM];
shared float s_w[QT * KT];
shared int s_rows[KT];
shared float s_max[QT];
shared float s_sum[QT];
shared float s_rescale[QT];

#include "block_sparse_common.glsl"

#if OPERAND_BF16
#define WIDEN(x) uintBitsToFloat(uint(x) << 16)
#else
#define WIDEN(x) (x)
#endif

float negativeInfinity() { return uintBitsToFloat(0xFF800000u); }

// The softmax weight of a score under a running maximum. A key a score of negative infinity excludes weighs nothing; a NaN
// score, or a maximum of positive infinity, gives NaN, which reaches the sum and so the output.
float weightOf(float score, float maximum) {
    if (floatBitsToUint(score) == 0xFF800000u) return 0.0;
    float difference = score - maximum;
    if (bsaIsNaN(score) || bsaIsNaN(difference)) return uintBitsToFloat(0x7FC00000u);
    return exp(difference);
}

void main() {
    uint queryLength = pc.qbt * pc.qbh * pc.qbw, keyLength = pc.kbt * pc.kbh * pc.kbw;
    uint tiles = (queryLength + QT - 1u) / QT;
    uint group = pc.firstGroup + gl_WorkGroupID.x;
    uint qb = group / tiles, tile = group % tiles;
    uint h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint tid = gl_LocalInvocationID.x;
    uint head = b * pc.heads + h;
    uint qRows = head * pc.queryTokens * pc.dim, kRows = head * pc.keyTokens * pc.dim;
    uint blocks = (head * pc.queryBlocks + qb) * pc.chosen;

    for (uint i = tid; i < QT * pc.dim; i += 128u) {
        uint query = i / pc.dim, d = i - query * pc.dim;
        uint offset = tile * QT + query;
        float value = 0.0;
        if (offset < queryLength)
            value = WIDEN(q_data[qRows + bsaRow(qb, offset, pc.qGridH, pc.qGridW, pc.qbt, pc.qbh, pc.qbw) * pc.dim + d]);
        s_q[i] = value;
    }
    if (tid < QT) {
        s_max[tid] = negativeInfinity();
        s_sum[tid] = 0.0;
    }

    uint queryPair = tid / 8u, keyPair = tid % 8u, channels = tid % 8u;
    uint q0 = queryPair * 2u, q1 = q0 + 1u;
    float acc0[16], acc1[16];
    for (uint j = 0u; j < 16u; ++j) {
        acc0[j] = 0.0;
        acc1[j] = 0.0;
    }

    uint total = pc.chosen * keyLength;
    for (uint start = 0u; start < total; start += KT) {
        barrier();
        if (tid < KT) {
            uint at = start + tid;
            int row = -1;
            if (at < total)
                row = int(bsaRow(uint(selected[blocks + at / keyLength]), at % keyLength, pc.kGridH, pc.kGridW, pc.kbt, pc.kbh, pc.kbw));
            s_rows[tid] = row;
        }
        barrier();
        for (uint i = tid; i < KT * pc.dim; i += 128u) {
            uint p = i / pc.dim, d = i - p * pc.dim;
            float value = 0.0;
            if (s_rows[p] >= 0) value = WIDEN(k_data[kRows + uint(s_rows[p]) * pc.dim + d]);
            s_kv[i] = value;
        }
        barrier();

        uint k0 = keyPair * 2u, k1 = k0 + 1u;
        float s00 = 0.0, s01 = 0.0, s10 = 0.0, s11 = 0.0;
        for (uint d = 0u; d < pc.dim; ++d) {
            float a = s_q[q0 * pc.dim + d], c = s_q[q1 * pc.dim + d];
            float x = s_kv[k0 * pc.dim + d], y = s_kv[k1 * pc.dim + d];
            s00 += a * x;
            s01 += a * y;
            s10 += c * x;
            s11 += c * y;
        }
        // A row past the keys is no key: negative infinity.
        bool live0 = s_rows[k0] >= 0, live1 = s_rows[k1] >= 0;
        s_w[q0 * KT + k0] = live0 ? s00 * pc.scale : negativeInfinity();
        s_w[q0 * KT + k1] = live1 ? s01 * pc.scale : negativeInfinity();
        s_w[q1 * KT + k0] = live0 ? s10 * pc.scale : negativeInfinity();
        s_w[q1 * KT + k1] = live1 ? s11 * pc.scale : negativeInfinity();
        barrier();

        if (tid < QT) {
            float chunkMax = negativeInfinity();
            for (uint k = 0u; k < KT; ++k) {
                float score = s_w[tid * KT + k];
                if (!bsaIsNaN(score) && score > chunkMax) chunkMax = score;
            }
            float runMax = s_max[tid];
            float newMax = chunkMax > runMax ? chunkMax : runMax;
            // Nothing seen before (a maximum of negative infinity) rescales an empty sum by zero.
            float rescale = newMax == runMax ? 1.0 : floatBitsToUint(runMax) == 0xFF800000u ? 0.0 : exp(runMax - newMax);
            float sum = 0.0;
            for (uint k = 0u; k < KT; ++k) {
                float weight = weightOf(s_w[tid * KT + k], newMax);
                s_w[tid * KT + k] = weight;
                sum += weight;
            }
            s_max[tid] = newMax;
            s_sum[tid] = s_sum[tid] * rescale + sum;
            s_rescale[tid] = rescale;
        }
        barrier();

        for (uint i = tid; i < KT * pc.dim; i += 128u) {
            uint p = i / pc.dim, d = i - p * pc.dim;
            float value = 0.0;
            if (s_rows[p] >= 0) value = WIDEN(v_data[kRows + uint(s_rows[p]) * pc.dim + d]);
            s_kv[i] = value;
        }
        barrier();

        float r0 = s_rescale[q0], r1 = s_rescale[q1];
        for (uint j = 0u; j < 16u; ++j) {
            acc0[j] *= r0;
            acc1[j] *= r1;
        }
        for (uint k = 0u; k < KT; ++k) {
            if (s_rows[k] < 0) break;
            float w0 = s_w[q0 * KT + k], w1 = s_w[q1 * KT + k];
            for (uint j = 0u; j < 16u; ++j) {
                uint d = channels * 16u + j;
                if (d >= pc.dim) break;
                float value = s_kv[k * pc.dim + d];
                acc0[j] += w0 * value;
                acc1[j] += w1 * value;
            }
        }
    }
    barrier();

    for (uint i = 0u; i < 2u; ++i) {
        uint query = i == 0u ? q0 : q1;
        uint offset = tile * QT + query;
        if (offset >= queryLength) continue;
        uint row = (head * pc.queryTokens + bsaRow(qb, offset, pc.qGridH, pc.qGridW, pc.qbt, pc.qbh, pc.qbw)) * pc.dim;
        // A query with no key is zero; a NaN sum is not zero and stays NaN.
        float sum = s_sum[query];
        bool empty = bsaIsZero(sum);
        for (uint j = 0u; j < 16u; ++j) {
            uint d = channels * 16u + j;
            if (d >= pc.dim) break;
            float result = empty ? 0.0 : (i == 0u ? acc0[j] : acc1[j]) / sum;
#if OUTPUT_BF16
            uint bits = floatBitsToUint(result);
            // NaN is written as the canonical quiet NaN: the rounding bias could carry its mantissa out of the exponent.
            uint narrow = (bits & 0x7FFFFFFFu) > 0x7F800000u ? 0x7FC0u : (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
            o_data[row + d] = uint16_t(narrow);
#else
            o_data[row + d] = result;
#endif
        }
    }
}
