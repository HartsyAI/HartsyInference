// sdpa_flash_tiled: query-tiled F32 flash attention with an online softmax. One workgroup per (query tile, head, batch).
// The query tile and each key/value tile are staged in shared memory; the score block and the output accumulator are
// per-thread registers. F32 throughout. Addressing is by element strides per tensor (Row, Head, Batch), so head-major and
// token-major are one kernel; the innermost dim is contiguous.
//
// Thread grid: NG = NT/8 row groups by 8 column groups. Row group rg owns query rows rg*RPT..; column group cg owns keys
// cg*KPT.. of each tile for the scores, and output dims cg, cg+8, .. for the accumulator. Every thread reaches every
// barrier: rows past sq and keys past the tile are masked, never skipped.
//
// Push constants: sdpa_flash's prefix (hq .. slidingWindow), the Q/K/V/O strides, then the bias strides under HAS_MASK.
// The bias is additive as in sdpa_flash_mask; a zero stride broadcasts that axis. Masks are the scalar kernel's.

#version 460

#ifndef MAX_D
#define MAX_D 128
#endif
#ifndef BR
#define BR 16
#endif
#ifndef TK
#define TK 32
#endif
#ifndef HAS_MASK
#define HAS_MASK 0
#endif

#ifndef NT
#define NT 64
#endif
#define NG (NT / 8)
#define RPT (BR / NG)
#define KPT (TK / 8)
#define DPT (MAX_D / 8)
#define SD (MAX_D + 1)
#define NEG_INF (uintBitsToFloat(0xff800000u))
#define SYNC() barrier(); memoryBarrierShared()
#define LPT ((MAX_D + NT - 1) / NT)
#ifndef LR
#define LR 8
#endif

layout(local_size_x = NT, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { float k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { float v_data[]; };
#if HAS_MASK == 1
layout(set = 0, binding = 3) readonly  buffer Mask_ { float mask_data[]; };
layout(set = 0, binding = 4) writeonly buffer O_ { float o_data[]; };
#else
layout(set = 0, binding = 3) writeonly buffer O_ { float o_data[]; };
#endif

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint sq;
    uint skv;
    uint kvCapacity;     // unused: K/V are addressed by the strides below; kept so the prefix matches sdpa_flash
    uint headDim;
    float scale;
    uint causal;         // 0/1
    uint qOffset;        // absolute position of query row 0
    uint slidingWindow;  // 0 = disabled; else max (qPos - kPos) distance
    uint qRow; uint qHead; uint qBatch;
    uint kRow; uint kHead; uint kBatch;
    uint vRow; uint vHead; uint vBatch;
    uint oRow; uint oHead; uint oBatch;
#if HAS_MASK == 1
    uint maskBatchStride; uint maskHeadStride; uint maskRowStride;
#endif
} pc;

shared float sQ[BR][SD];
shared float sK[TK][SD];
shared float sV[TK][SD];
shared float sP[BR][TK + 1];
shared float sRedMax[BR][8];
shared float sRedSum[BR][8];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint rg = tid / 8u;
    uint cg = tid % 8u;
    uint q0 = gl_WorkGroupID.x * uint(BR);
    uint h = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint D = pc.headDim;
    uint kvHead = h / (pc.hq / pc.hkv);

    uint qBase = b * pc.qBatch + h * pc.qHead;
    uint kBase = b * pc.kBatch + kvHead * pc.kHead;
    uint vBase = b * pc.vBatch + kvHead * pc.vHead;
    uint oBase = b * pc.oBatch + h * pc.oHead;

    for (uint r = 0; r < uint(BR); r++) {
        uint qi = q0 + r;
        for (uint d = tid; d < D; d += uint(NT)) {
            sQ[r][d] = (qi < pc.sq) ? q_data[qBase + qi * pc.qRow + d] : 0.0;
        }
    }
    SYNC();

    float runMax[RPT];
    float runSum[RPT];
    float acc[RPT][DPT];
    [[unroll]] for (int i = 0; i < RPT; i++) {
        runMax[i] = NEG_INF;
        runSum[i] = 0.0;
        [[unroll]] for (int jj = 0; jj < DPT; jj++) acc[i][jj] = 0.0;
    }

    for (uint kStart = 0; kStart < pc.skv; kStart += uint(TK)) {
        uint tileLen = min(uint(TK), pc.skv - kStart);
        // LR rows at a time, every load of a batch issued before its first store: one memory stall per batch, not per element.
        for (uint r0 = 0; r0 < uint(TK); r0 += uint(LR)) {
            float kr[LR][LPT];
            float vr[LR][LPT];
            [[unroll]] for (int rr = 0; rr < LR; rr++) {
                uint r = r0 + uint(rr);
                uint kt = kStart + r;
                bool valid = r < tileLen;
                [[unroll]] for (int j = 0; j < LPT; j++) {
                    uint d = tid + uint(j * NT);
                    bool ok = valid && d < D;
                    kr[rr][j] = ok ? k_data[kBase + kt * pc.kRow + d] : 0.0;
                    vr[rr][j] = ok ? v_data[vBase + kt * pc.vRow + d] : 0.0;
                }
            }
            [[unroll]] for (int rr = 0; rr < LR; rr++) {
                [[unroll]] for (int j = 0; j < LPT; j++) {
                    uint d = tid + uint(j * NT);
                    if (d < uint(MAX_D)) {
                        sK[r0 + uint(rr)][d] = kr[rr][j];
                        sV[r0 + uint(rr)][d] = vr[rr][j];
                    }
                }
            }
        }
        SYNC();

        float s[RPT][KPT];
        [[unroll]] for (int i = 0; i < RPT; i++) {
            [[unroll]] for (int j = 0; j < KPT; j++) s[i][j] = 0.0;
        }
        for (uint d = 0; d < D; d++) {
            float qv[RPT];
            [[unroll]] for (int i = 0; i < RPT; i++) qv[i] = sQ[rg * uint(RPT) + uint(i)][d];
            [[unroll]] for (int j = 0; j < KPT; j++) {
                float kv = sK[cg * uint(KPT) + uint(j)][d];
                [[unroll]] for (int i = 0; i < RPT; i++) s[i][j] += qv[i] * kv;
            }
        }

        [[unroll]] for (int i = 0; i < RPT; i++) {
            uint qi = q0 + rg * uint(RPT) + uint(i);
            uint absQ = pc.qOffset + qi;
            [[unroll]] for (int j = 0; j < KPT; j++) {
                uint kLocal = cg * uint(KPT) + uint(j);
                uint kt = kStart + kLocal;
                bool masked = kLocal >= tileLen || qi >= pc.sq
                    || (pc.causal != 0u && kt > absQ)
                    || (pc.slidingWindow != 0u && kt + pc.slidingWindow <= absQ);
                float v = masked ? NEG_INF : s[i][j] * pc.scale;
#if HAS_MASK == 1
                if (!masked) {
                    v += mask_data[b * pc.maskBatchStride + h * pc.maskHeadStride + qi * pc.maskRowStride + kt];
                }
#endif
                s[i][j] = v;
            }
        }

        [[unroll]] for (int i = 0; i < RPT; i++) {
            float m = NEG_INF;
            [[unroll]] for (int j = 0; j < KPT; j++) m = max(m, s[i][j]);
            sRedMax[rg * uint(RPT) + uint(i)][cg] = m;
        }
        SYNC();

        // Every thread of a row group reads the same eight partials, so each keeps its own running max and sum.
        float corr[RPT];
        [[unroll]] for (int i = 0; i < RPT; i++) {
            uint row = rg * uint(RPT) + uint(i);
            float tileMax = NEG_INF;
            [[unroll]] for (uint c = 0; c < 8u; c++) tileMax = max(tileMax, sRedMax[row][c]);
            float newMax = max(runMax[i], tileMax);
            // A row with no valid key yet keeps P = 0 and its accumulator: exp(-inf - -inf) would be NaN.
            bool anyValid = newMax > NEG_INF;
            corr[i] = anyValid ? exp(runMax[i] - newMax) : 0.0;
            float ls = 0.0;
            [[unroll]] for (int j = 0; j < KPT; j++) {
                float p = anyValid ? exp(s[i][j] - newMax) : 0.0;
                s[i][j] = p;
                ls += p;
            }
            runMax[i] = newMax;
            sRedSum[row][cg] = ls;
            [[unroll]] for (int j = 0; j < KPT; j++) sP[row][cg * uint(KPT) + uint(j)] = s[i][j];
        }
        SYNC();

        [[unroll]] for (int i = 0; i < RPT; i++) {
            uint row = rg * uint(RPT) + uint(i);
            float tileSum = 0.0;
            [[unroll]] for (uint c = 0; c < 8u; c++) tileSum += sRedSum[row][c];
            runSum[i] = runSum[i] * corr[i] + tileSum;
            [[unroll]] for (int jj = 0; jj < DPT; jj++) acc[i][jj] *= corr[i];
        }
        for (uint c = 0; c < tileLen; c++) {
            float pv[RPT];
            [[unroll]] for (int i = 0; i < RPT; i++) pv[i] = sP[rg * uint(RPT) + uint(i)][c];
            [[unroll]] for (int jj = 0; jj < DPT; jj++) {
                uint d = cg + 8u * uint(jj);
                if (d < D) {
                    float vv = sV[c][d];
                    [[unroll]] for (int i = 0; i < RPT; i++) acc[i][jj] += pv[i] * vv;
                }
            }
        }
        SYNC();
    }

    [[unroll]] for (int i = 0; i < RPT; i++) {
        uint qi = q0 + rg * uint(RPT) + uint(i);
        if (qi < pc.sq) {
            float inv = runSum[i] > 0.0 ? 1.0 / runSum[i] : 0.0;
            [[unroll]] for (int jj = 0; jj < DPT; jj++) {
                uint d = cg + 8u * uint(jj);
                if (d < D) o_data[oBase + qi * pc.oRow + d] = acc[i][jj] * inv;
            }
        }
    }
}
