// sparse_latent_attention: one workgroup per (token, head), matching SparseLatentAttentionReference.Apply. One latent row
// is both key and value; per head, softmax over the scaled dots of the valid indexed rows plus the head's sink (which joins
// the denominator only), then the probability-weighted row sum. Index i addresses the window ring below windowSlots and the
// main cache after it; -1 and anything past both sources are skipped.
//
// Each dot and each output sum is accumulated serially in the reference's order with fusing disabled, so only exp differs
// from the reference. Shared memory holds the k probabilities (MAX_K x 4 bytes; with the reduction scratch that stays under the 16 KB every device guarantees).
// WGSIZE must equal the dispatch's local_size_x.

#version 460
#extension GL_GOOGLE_include_directive : require

#define WGSIZE 256
#define MAX_K 3800

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Out_ { float out_data[]; };
layout(set = 0, binding = 1) readonly buffer Query_ { float q_data[]; };
layout(set = 0, binding = 2) readonly buffer WCodes_ { uint w_codes[]; };
layout(set = 0, binding = 3) readonly buffer WScales_ { uint w_scales[]; };
layout(set = 0, binding = 4) readonly buffer MCodes_ { uint m_codes[]; };
layout(set = 0, binding = 5) readonly buffer MScales_ { uint m_scales[]; };
layout(set = 0, binding = 6) readonly buffer Indices_ { int idx_data[]; };
layout(set = 0, binding = 7) readonly buffer Sink_ { float sink_data[]; };

layout(push_constant) uniform Push {
    uint pairs;          // tokens * heads
    uint heads;
    uint dim;
    uint k;
    uint windowSlots;
    uint mainRows;
    uint windowEnc;
    uint mainEnc;
    float scale;
    uint pairBase;
} pc;

#include "div_rn.glsl"
#include "latent_codec.glsl"

LATENT_DECODE_FN(windowAt, w_codes, w_scales)
LATENT_DECODE_FN(mainAt, m_codes, m_scales)

// exp accurate to about one ulp on every driver (see softplus.comp.glsl): hand range reduction plus a degree-7 series.
float accExp(float x) {
    if (x > 88.0) return uintBitsToFloat(0x7F800000u);
    if (x < -87.0) return 0.0;
    float kf = floor(x * 1.44269504089 + 0.5);
    float r = fma(-kf, 0.693145751953125, x);
    r = fma(-kf, 1.42860677e-6, r);
    float p = 1.0 + r * (1.0 + r * 0.5 * (1.0 + r * (1.0 / 3.0) * (1.0 + r * 0.25 * (1.0 + r * 0.2
        * (1.0 + r * (1.0 / 6.0) * (1.0 + r * (1.0 / 7.0)))))));
    return ldexp(p, int(kf));
}

shared float sProb[MAX_K];
shared float sRed[WGSIZE];
shared float sMax;
shared float sDenom;

bool validId(int id) {
    return id >= 0 && uint(id) < pc.windowSlots + pc.mainRows;
}

float rowAt(uint id, uint d) {
    if (id < pc.windowSlots) return windowAt(pc.windowEnc, id, pc.dim, d);
    return mainAt(pc.mainEnc, id - pc.windowSlots, pc.dim, d);
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint pair = pc.pairBase + gl_WorkGroupID.x;
    if (pair >= pc.pairs) return;
    uint t = pair / pc.heads;
    uint qBase = pair * pc.dim;
    uint idxBase = t * pc.k;
    float sink = sink_data[pair - t * pc.heads];

    float localMax = sink;
    for (uint j = tid; j < pc.k; j += WGSIZE) {
        int id = idx_data[idxBase + j];
        if (!validId(id)) continue;
        precise float dot = 0.0;
        for (uint d = 0u; d < pc.dim; d++) {
            dot += q_data[qBase + d] * rowAt(uint(id), d);
        }
        precise float logit = dot * pc.scale;
        sProb[j] = logit;
        localMax = max(localMax, logit);
    }
    sRed[tid] = localMax;
    barrier();
    for (uint s = WGSIZE / 2u; s > 0u; s >>= 1) {
        if (tid < s) sRed[tid] = max(sRed[tid], sRed[tid + s]);
        barrier();
    }
    if (tid == 0u) sMax = sRed[0];
    barrier();
    float mx = sMax;

    for (uint j = tid; j < pc.k; j += WGSIZE) {
        if (validId(idx_data[idxBase + j])) sProb[j] = accExp(sProb[j] - mx);
    }
    barrier();
    if (tid == 0u) {
        precise float denom = accExp(sink - mx);
        for (uint j = 0u; j < pc.k; j++) {
            if (validId(idx_data[idxBase + j])) denom += sProb[j];
        }
        sDenom = denom;
    }
    barrier();
    float denom = sDenom;

    for (uint d = tid; d < pc.dim; d += WGSIZE) {
        precise float acc = 0.0;
        for (uint j = 0u; j < pc.k; j++) {
            int id = idx_data[idxBase + j];
            if (validId(id)) acc += sProb[j] * rowAt(uint(id), d);
        }
        out_data[qBase + d] = divRn(acc, denom);
    }
}
