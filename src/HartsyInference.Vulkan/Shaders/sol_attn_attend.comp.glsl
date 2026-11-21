// General Sol-Attn in F32, stage 5 of 5 (the token pass). SolAttentionReference in Core defines it; sol_attn_f32.cu is the
// CUDA form. Operands are token-major [B, T, H, D]; tokens are cut into blocks of 64.
//
// Per (query token, head, batch): one base-2 softmax over the exactly read keys and, when the tail is on, one pooled
// entry per unread block weighing its live length. Invocation j is token j of whichever key block is being read.
// The bias is in natural-log units, [T] (biasStride 0) or [B, T] (biasStride T); -inf masks a key. A buffer is always
// bound for it; hasBias says whether it is read. Ordinary F32 arithmetic: this stage is compared by tolerance.

#version 460

layout(local_size_x = 64) in;

#define NEG_INF (-1.0 / 0.0)
#define LOG2E 1.4426950408889634

layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { float k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { float v_data[]; };
layout(set = 0, binding = 3) readonly  buffer M_ { float key_mean[]; };
layout(set = 0, binding = 4) readonly  buffer P_ { float pooled[]; };
layout(set = 0, binding = 5) readonly  buffer VS_ { float value_sums[]; };
layout(set = 0, binding = 6) readonly  buffer R_ { uint reads[]; };
layout(set = 0, binding = 7) readonly  buffer Len_ { int len_data[]; };
layout(set = 0, binding = 8) readonly  buffer B_ { float bias_data[]; };
layout(set = 0, binding = 9) writeonly buffer O_ { float o_data[]; };

layout(push_constant) uniform Push {
    uint tokens;
    uint heads;
    uint dim;
    uint blocks;
    float log2s;
    uint tail;
    uint biasStride;
    uint hasBias;
} pc;

shared float s_q[256];
shared float s_p[64];

uint row(uint b, uint t, uint h) { return ((b * pc.tokens + t) * pc.heads + h) * pc.dim; }

// The logit of token tid of key block n, or -inf when that row is not live or is masked.
float logit(uint n, uint tid, uint b, uint h, uint mean) {
    if (int(tid) >= len_data[n]) return NEG_INF;
    uint j = n * 64u + tid;
    uint k = row(b, j, h);
    float dot = 0.0;
    for (uint d = 0u; d < pc.dim; ++d) dot += s_q[d] * (k_data[k + d] - key_mean[mean + d]);
    float value = dot * pc.log2s;
    if (pc.hasBias != 0u) value += bias_data[b * pc.biasStride + j] * LOG2E;
    return value;
}

void main() {
    uint i = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint tid = gl_LocalInvocationID.x;
    uint qb = i / 64u;
    uint head = b * pc.heads + h;
    uint qAt = row(b, i, h);
    uint mean = head * pc.dim;
    uint scores = (head * pc.blocks + qb) * pc.blocks;
    for (uint d = tid; d < pc.dim; d += 64u) s_q[d] = q_data[qAt + d];
    barrier();

    float m = NEG_INF;
    for (uint n = 0u; n < pc.blocks; ++n) {
        if (reads[scores + n] != 0u) m = max(m, logit(n, tid, b, h, mean));
        else if (pc.tail != 0u && tid == 0u) m = max(m, pooled[scores + n]);
    }
    s_p[tid] = m;
    barrier();
    float gmax = NEG_INF;
    for (uint j = 0u; j < 64u; ++j) gmax = max(gmax, s_p[j]);
    barrier();

    float acc[4] = float[4](0.0, 0.0, 0.0, 0.0);
    float denominator = 0.0;
    if (gmax > NEG_INF) {
        for (uint n = 0u; n < pc.blocks; ++n) {
            if (reads[scores + n] != 0u) {
                float value = logit(n, tid, b, h, mean);
                s_p[tid] = value > NEG_INF ? exp2(value - gmax) : 0.0;
                barrier();
                uint len = uint(len_data[n]);
                for (uint j = 0u; j < len; ++j) denominator += s_p[j];
                for (uint e = 0u; e < 4u; ++e) {
                    uint d = tid + e * 64u;
                    if (d >= pc.dim) break;
                    float sum = 0.0;
                    for (uint j = 0u; j < len; ++j) sum += s_p[j] * v_data[row(b, n * 64u + j, h) + d];
                    acc[e] += sum;
                }
                barrier();
            } else if (pc.tail != 0u) {
                float p = exp2(pooled[scores + n] - gmax);
                denominator += p * float(len_data[n]);
                uint sums = (head * pc.blocks + n) * pc.dim;
                for (uint e = 0u; e < 4u; ++e) {
                    uint d = tid + e * 64u;
                    if (d >= pc.dim) break;
                    acc[e] += p * value_sums[sums + d];
                }
            }
        }
    }
    float inverse = 1.0 / max(denominator, 1e-30);
    for (uint e = 0u; e < 4u; ++e) {
        uint d = tid + e * 64u;
        if (d >= pc.dim) break;
        o_data[qAt + d] = acc[e] * inverse;
    }
}
