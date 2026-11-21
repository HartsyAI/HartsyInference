// int8_attn_u8_tile: the first pass of INT8 attention with u8 probabilities, as the recurrence of
// Int8AttentionV2Reference takes one key tile for one query row, unmasked and with per-key scales. For each (row, tile):
//   score[j] = fma(q8 . k8[j], (qScale * scaleLog2e) * kScale[j], shift)   the dot a whole number below 2^24
//   tileMax  = the largest score, a NaN if any score is one
//   P[j]     = roundEven(clamp(2^(score[j] - (tileMax - offset)), 0, 255)), so the tile's largest score maps to 255
//   sum      = the total of the tile's P bytes
// shift is (meanDot * scaleLog2e) minus itself: zero, or a NaN where the query's dot with the key mean is not finite,
// which is what the reference adds once the first key segment's offset is left out of the scores.
// A tile whose maximum is not finite gets no P bytes; the drain pass turns the row into NaNs.
//
// Q and K are signed bytes, four to an int word in channel order. P is written four bytes to a uint word in key order,
// each tile padded to a whole tile of key slots. One invocation per (row, tile).
//
// The multiply-add goes through fmaRn (fma_rn.glsl) and 2^x through exp2Fixed (exp2_fixed.glsl, whose host twin is
// Int8AttentionExp2.Evaluate), so the bytes are the reference's on any device.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer Q_  { int q[]; };
layout(set = 0, binding = 1) readonly  buffer K_  { int k[]; };
layout(set = 0, binding = 2) readonly  buffer QS_ { float qScale[]; };
layout(set = 0, binding = 3) readonly  buffer KS_ { float kScale[]; };
layout(set = 0, binding = 4) readonly  buffer MD_ { float meanDot[]; };
layout(set = 0, binding = 5) writeonly buffer TM_ { float tileMax[]; };
layout(set = 0, binding = 6) writeonly buffer SM_ { uint tileSum[]; };
layout(set = 0, binding = 7) writeonly buffer P_  { uint pw[]; };

layout(push_constant) uniform Push {
    uint rowsTotal;     // heads * rows
    uint rowsPerHead;
    uint group;         // query heads a KV head
    uint keys;
    uint K;             // padded head dim, a multiple of 4
    uint bc;            // keys a tile, 64 or 128
    uint tiles;
    float scaleLog2e;   // the call's scale times log2(e)
    float offset;       // 7.9943534: the tile maximum maps to 255
    uint base;          // the first (row, tile) of this dispatch
} pc;

#include "fma_rn.glsl"
#include "exp2_fixed.glsl"

int dot4(int a, int b) {
    return ((a << 24) >> 24) * ((b << 24) >> 24) + ((a << 16) >> 24) * ((b << 16) >> 24)
         + ((a << 8) >> 24) * ((b << 8) >> 24) + (a >> 24) * (b >> 24);
}

void main() {
    uint i = pc.base + gl_GlobalInvocationID.x;
    if (i >= pc.rowsTotal * pc.tiles) return;
    uint row = i / pc.tiles, t = i - row * pc.tiles;
    uint kvHead = (row / pc.rowsPerHead) / pc.group;
    uint j0 = t * pc.bc, j1 = min(j0 + pc.bc, pc.keys);
    uint words = pc.K / 4u, qBase = row * words;

    precise float qs = qScale[row] * pc.scaleLog2e;
    precise float md = meanDot[row] * pc.scaleLog2e;
    // md - md, written out so that no compiler may take it for zero when md is not finite.
    float shift = (isnan(md) || isinf(md)) ? uintBitsToFloat(0x7FC00000u) : 0.0;

    float score[128];   // one a key of the tile: bc is 64 or 128, which the host entry enforces
    float best = uintBitsToFloat(0xFF800000u);
    bool anyNan = false;
    for (uint j = j0; j < j1; j++) {
        uint kRow = kvHead * pc.keys + j, kBase = kRow * words;
        int acc = 0;
        for (uint w = 0u; w < words; w++) acc += dot4(q[qBase + w], k[kBase + w]);
        precise float s = qs * kScale[kRow];
        float x = fmaRn(float(acc), s, shift);
        score[j - j0] = x;
        if (isnan(x)) anyNan = true;
        else best = max(best, x);
    }
    if (anyNan) best = uintBitsToFloat(0x7FC00000u);
    tileMax[i] = best;

    uint pBase = (row * pc.tiles + t) * (pc.bc / 4u);
    uint total = 0u;
    bool finite = !anyNan && !isinf(best);
    precise float tileM = best - pc.offset;
    for (uint w = 0u; w < pc.bc / 4u; w++) {
        uint word = 0u;
        for (uint b = 0u; b < 4u; b++) {
            uint j = j0 + w * 4u + b;
            if (!finite || j >= j1) continue;
            precise float d = score[j - j0] - tileM;
            float x = exp2Fixed(d);
            uint p = isnan(x) ? 0u : uint(roundEven(clamp(x, 0.0, 255.0)));
            word |= p << (8u * b);
            total += p;
        }
        pw[pBase + w] = word;
    }
    tileSum[i] = total;
}
