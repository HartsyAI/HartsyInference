// sdpa_flash: fused online-softmax scaled-dot-product attention. Never materializes the [Sq,Skv] score
// matrix — the documented root cause of the Wan-video full-resolution OOM on the old 3-pass path (a
// single self-attention call there needs a ~19 GB score matrix). One workgroup per (batch, head, query
// row); each of the TILE threads in the workgroup owns one key within the current KV tile, so scoring a
// tile needs no cross-thread reduction (only tile-level online-softmax combine, done by thread 0 over
// TILE values). GQA: query head h reads kv head h/(hq/hkv). Supports causal masking, a sliding window,
// and (HAS_MASK=1 variant) an optional additive [Sq,Skv] mask broadcast across batch/head — the same
// broadcast convention VulkanBackend's prior 3-pass SDPA used. Does NOT support softcap/sink/ALiBi
// (Gemma-2/GPT-OSS/MPT-class models) — those fall through to IBackend's CPU reference; a documented
// "flash-lite" scope boundary, not an oversight (see docs/Checklists/TROUBLESHOOTING.md).
//
// Algorithm per query row (standard tiled online-softmax flash attention, Br=1):
//   m = -inf, l = 0, acc[d] = 0
//   for each KV tile:
//     load K/V tile into shared memory (one thread per key)
//     score[tid] = dot(Q, K[tid]) * scale (+ mask), or -inf if causal/window-masked
//     tileMax = max over tile; newMax = max(m, tileMax)
//     corrOld = exp(m - newMax); p[j] = exp(score[j] - newMax); tileSum = sum(p)
//     l = l*corrOld + tileSum
//     acc[d] = acc[d]*corrOld + sum_j(p[j] * V[j][d])   (each thread owns a strided subset of d)
//     m = newMax
//   O = acc / l
//
// Compile:
//   glslc sdpa_flash.comp.glsl -o sdpa_flash_f32.spv
//   glslc -DUSE_FP16=1 sdpa_flash.comp.glsl -o sdpa_flash_f16.spv
//   glslc -DHAS_MASK=1 sdpa_flash.comp.glsl -o sdpa_flash_mask_f32.spv
//   glslc -DUSE_FP16=1 -DHAS_MASK=1 sdpa_flash.comp.glsl -o sdpa_flash_mask_f16.spv

#version 460

#ifndef USE_FP16
#define USE_FP16 0
#endif
#ifndef HAS_MASK
#define HAS_MASK 0
#endif

#if USE_FP16 == 1
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#define DTYPE float16_t
#define TO_F32(x) float(x)
#define FROM_F32(x) float16_t(x)
#else
#define DTYPE float
#define TO_F32(x) (x)
#define FROM_F32(x) (x)
#endif

// TILE = keys processed per iteration = threads per workgroup (one thread per key in the tile).
// MAX_D = compile-time upper bound on head dim for the shared-memory tiles; real head dims (64-160)
// must fit. TILE*MAX_D*4 bytes/array * 2 arrays (K,V) = 32*128*4*2 = 32 KiB, safely under the common
// 48 KiB shared-memory budget.
#define TILE 32
#define MAX_D 128

layout(local_size_x = TILE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer Q_ { DTYPE q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { DTYPE k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { DTYPE v_data[]; };
#if HAS_MASK == 1
layout(set = 0, binding = 3) readonly  buffer Mask_ { float mask_data[]; };
layout(set = 0, binding = 4) writeonly buffer O_ { DTYPE o_data[]; };
#else
layout(set = 0, binding = 3) writeonly buffer O_ { DTYPE o_data[]; };
#endif

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint sq;
    uint skv;            // number of VALID kv positions to attend over (the loop bound)
    uint kvCapacity;      // actual per-head stride in the K/V buffer — may exceed skv when the buffer
                          // is an over-allocated KV cache (only the first `skv` positions are valid)
    uint headDim;
    float scale;
    uint causal;         // 0/1
    uint qOffset;        // absolute position of query row 0 (nonzero for a decode step appending to a KV cache)
    uint slidingWindow;  // 0 = disabled; else max (qPos - kPos) distance
} pc;

shared float sQ[MAX_D];
shared float sK[TILE][MAX_D];
shared float sV[TILE][MAX_D];
shared float sScore[TILE];
shared float sRowMax;
shared float sCorrOld;
shared float sRowSum;
shared float sAcc[MAX_D];

void main() {
    uint queryRow = gl_WorkGroupID.x;
    uint hIdx = gl_WorkGroupID.y;
    uint b = gl_WorkGroupID.z;
    uint tid = gl_LocalInvocationID.x;

    uint kvGroup = pc.hq / pc.hkv;
    uint kvHeadIdx = hIdx / kvGroup;
    uint D = pc.headDim;

    uint qBase = ((b * pc.hq + hIdx) * pc.sq + queryRow) * D;

    for (uint d = tid; d < D; d += TILE) { sQ[d] = TO_F32(q_data[qBase + d]); sAcc[d] = 0.0; }
    if (tid == 0) { sRowMax = -1.0 / 0.0; sRowSum = 0.0; }
    barrier();
    memoryBarrierShared();

    uint absQPos = pc.qOffset + queryRow;

    for (uint kvStart = 0; kvStart < pc.skv; kvStart += TILE) {
        uint tileLen = min(uint(TILE), pc.skv - kvStart);

        if (tid < tileLen) {
            uint keyIdx = kvStart + tid;
            uint kBase = ((b * pc.hkv + kvHeadIdx) * pc.kvCapacity + keyIdx) * D;
            for (uint d = 0; d < D; d++) {
                sK[tid][d] = TO_F32(k_data[kBase + d]);
                sV[tid][d] = TO_F32(v_data[kBase + d]);
            }
        }
        barrier();
        memoryBarrierShared();

        if (tid < tileLen) {
            uint keyIdx = kvStart + tid;
            bool masked = false;
            if (pc.causal != 0 && keyIdx > absQPos) masked = true;
            if (pc.slidingWindow != 0 && (keyIdx > absQPos || absQPos - keyIdx >= pc.slidingWindow)) masked = true;
            if (masked) {
                sScore[tid] = -1.0 / 0.0;
            } else {
                float dot = 0.0;
                for (uint d = 0; d < D; d++) dot += sQ[d] * sK[tid][d];
                float s = dot * pc.scale;
#if HAS_MASK == 1
                s += mask_data[queryRow * pc.skv + keyIdx];
#endif
                sScore[tid] = s;
            }
        } else if (tid < TILE) {
            sScore[tid] = -1.0 / 0.0;
        }
        barrier();
        memoryBarrierShared();

        if (tid == 0) {
            float tileMax = -1.0 / 0.0;
            for (uint j = 0; j < tileLen; j++) tileMax = max(tileMax, sScore[j]);
            float newMax = max(sRowMax, tileMax);
            // Guard the degenerate "nothing valid seen yet" case (newMax == -inf, e.g. a sliding window
            // whose first admissible tile hasn't been reached): exp(-inf - -inf) is NaN, not 0.
            bool anyValid = newMax > -1.0 / 0.0;
            float corrOld = anyValid ? exp(sRowMax - newMax) : 0.0;
            float tileSum = 0.0;
            for (uint j = 0; j < tileLen; j++) {
                float p = anyValid ? exp(sScore[j] - newMax) : 0.0;
                sScore[j] = p;
                tileSum += p;
            }
            sRowSum = sRowSum * corrOld + tileSum;
            sCorrOld = corrOld;
            sRowMax = newMax;
        }
        barrier();
        memoryBarrierShared();

        for (uint d = tid; d < D; d += TILE) {
            float sum = 0.0;
            for (uint j = 0; j < tileLen; j++) sum += sScore[j] * sV[j][d];
            sAcc[d] = sAcc[d] * sCorrOld + sum;
        }
        barrier();
        memoryBarrierShared();
    }

    float invSum = sRowSum > 0.0 ? 1.0 / sRowSum : 0.0;
    for (uint d = tid; d < D; d += TILE) {
        o_data[qBase + d] = FROM_F32(sAcc[d] * invSum);
    }
}
