// int8_attn_u8_drain: the second pass of INT8 attention with u8 probabilities. It carries the reference's online
// recurrence over the key tiles of one query row for one output channel, from what int8_attn_u8_tile wrote:
//   tileM = tileMax - offset
//   first tile:   m = tileM, both scales 1
//   later tiles:  smaller = 2^-|m - tileM|; oScale = m < tileM ? smaller : 1; tileScale = tileM < m ? smaller : 1;
//                 m = max(m, tileM)
//   partial = sum over the tile's keys of P[j] * v8[j][c]              a whole number below 2^24
//   o = fma(partial, tileScale (* the tile's V scale), o * oScale)
//   d = fma(sum, tileScale, d * oScale)
// and at the end o / d, times the head's V scale where the scales are not per tile. A tile whose maximum is not finite
// makes the row NaN; a row with no keys is zero.
//
// V is signed bytes, four to an int word in channel order; P is four bytes to a uint word in key order, each tile
// padded to a whole tile. One invocation per (row, output channel).
//
// Each fused multiply-add goes through fmaRn, 2^x through exp2Fixed and the divide through divRn, so the output is the
// reference's bit for bit on any device, with one exception: where the reference itself passes through a subnormal
// (a tile scale times a V scale below 2^-126, or a carried sum that small), a device that reads subnormals as zero
// drops that tile's term. The term is at most 128 * 255 * 127 * 2^-126, about 2^-104, so only an output that is
// itself that small can show it, with one further case: a carried sum the reference holds as a subnormal still
// decides an exact rounding tie of the next drain by its sign, and here it is zero, so that output may differ in its
// last place. Nothing has been seen to hit it.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer P_  { uint pw[]; };
layout(set = 0, binding = 1) readonly  buffer V_  { int v[]; };
layout(set = 0, binding = 2) readonly  buffer VS_ { float vScale[]; };
layout(set = 0, binding = 3) readonly  buffer TM_ { float tileMax[]; };
layout(set = 0, binding = 4) readonly  buffer SM_ { uint tileSum[]; };
layout(set = 0, binding = 5) writeonly buffer O_  { float o[]; };

layout(push_constant) uniform Push {
    uint rowsTotal;     // heads * rows
    uint rowsPerHead;
    uint group;         // query heads a KV head
    uint keys;
    uint K;             // padded head dim
    uint bc;            // keys a tile
    uint tiles;
    uint outDim;        // output channels a row, at most K
    uint perTile;       // 1: one V scale a channel for each tile; 0: one for the head
    float offset;       // 7.9943534
    uint base;          // the first (row, channel) of this dispatch
} pc;

#include "fma_rn.glsl"
#include "exp2_fixed.glsl"
#include "div_rn.glsl"

void main() {
    uint i = pc.base + gl_GlobalInvocationID.x;
    if (i >= pc.rowsTotal * pc.outDim) return;
    uint row = i / pc.outDim, c = i - row * pc.outDim;
    uint kvHead = (row / pc.rowsPerHead) / pc.group;
    uint words = pc.K / 4u, shiftV = 24u - 8u * (c & 3u);

    bool valid = false, poisoned = false;
    float m = 0.0, d = 0.0, acc = 0.0;
    for (uint t = 0u; t < pc.tiles; t++) {
        float tm = tileMax[row * pc.tiles + t];
        if (isnan(tm) || isinf(tm)) { poisoned = true; break; }
        precise float tileM = tm - pc.offset;
        float oScale = 1.0, tileScale = 1.0;
        if (!valid) {
            m = tileM;
            valid = true;
        } else {
            precise float gap = m - tileM;
            float smaller = exp2Fixed(-abs(gap));
            oScale = m < tileM ? smaller : 1.0;
            tileScale = tileM < m ? smaller : 1.0;
            m = max(m, tileM);
        }
        uint j0 = t * pc.bc, j1 = min(j0 + pc.bc, pc.keys);
        uint pBase = (row * pc.tiles + t) * (pc.bc / 4u);
        int partial = 0;
        for (uint j = j0; j < j1; j++) {
            uint slot = j - j0;
            uint p = (pw[pBase + slot / 4u] >> (8u * (slot & 3u))) & 0xFFu;
            if (p == 0u) continue;
            int word = v[(kvHead * pc.keys + j) * words + c / 4u];
            partial += int(p) * ((word << shiftV) >> 24);
        }
        float scale = tileScale;
        if (pc.perTile != 0u) {
            precise float scaled = tileScale * vScale[(kvHead * pc.tiles + t) * pc.K + c];
            scale = scaled;
        }
        precise float carried = acc * oScale;
        acc = fmaRn(float(partial), scale, carried);
        precise float carriedD = d * oScale;
        d = fmaRn(float(tileSum[row * pc.tiles + t]), tileScale, carriedD);
    }
    if (poisoned) {
        o[i] = uintBitsToFloat(0x7FC00000u);
        return;
    }
    if (!valid) {
        o[i] = 0.0;
        return;
    }
    precise float inv = divRn(1.0, d);
    precise float result = acc * inv;
    if (pc.perTile == 0u) result = result * vScale[kvHead * pc.K + c];
    o[i] = result;
}
