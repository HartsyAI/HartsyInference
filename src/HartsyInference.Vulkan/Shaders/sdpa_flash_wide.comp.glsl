// sdpa_flash_wide: F32 online-softmax attention for the unmasked FlashAttention calls sdpa_flash cannot take: head dims
// up to MAX_D (1024, past sdpa_flash's 128) and the logit terms it lacks, a soft-cap and per-head sinks
// (docs/ATTENTION_CONTRACT.md §3). The HCA and DSA single-query MQA calls reach it through AttentionFlashLowering.
//
// One workgroup per (query row, head, batch). The query row sits in shared memory. Keys are scored a TILE at a time,
// one key per thread (a whole dot product each); a tile's max and exp-sum are workgroup reductions that update the running
// max m and denominator l. Each thread owns the output columns t, t + TILE, ... and accumulates p·V over the tile's keys,
// reading V rows across threads. A sink joins the denominator only: m starts at sink[h] and l at 1, else -inf and 0.
// GQA: query head h reads key/value head h / (hq / hkv). Keys [0, skv) are all visible; skv >= 1.
//
// Bindings: 0=Q [B,Hq,Sq,D], 1=K [B,Hkv,cap,D], 2=V [B,Hkv,cap,D], 3=sink [Hq] (unread without one), 4=O [B,Hq,Sq,D]
#version 460

#define TILE 128
#define MAX_D 1024
#define COLS (MAX_D / TILE)
#define NEG_INF uintBitsToFloat(0xff800000u)

layout(local_size_x = TILE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer Q_ { float q_data[]; };
layout(set = 0, binding = 1) readonly  buffer K_ { float k_data[]; };
layout(set = 0, binding = 2) readonly  buffer V_ { float v_data[]; };
layout(set = 0, binding = 3) readonly  buffer S_ { float sink_data[]; };
layout(set = 0, binding = 4) writeonly buffer O_ { float o_data[]; };

layout(push_constant) uniform Push {
    uint hq;
    uint hkv;
    uint sq;
    uint skv;
    uint kvCapacity;
    uint headDim;
    float scale;
    float softcap;
    uint hasSink;
} pc;

shared float qs[MAX_D];
shared float ps[TILE];
shared float red[TILE];

// Workgroup max (isMax) or sum of v; every invocation must call it.
float reduce(float v, bool isMax) {
    uint t = gl_LocalInvocationID.x;
    red[t] = v;
    barrier();
    for (uint s = TILE / 2u; s > 0u; s >>= 1u) {
        if (t < s) red[t] = isMax ? max(red[t], red[t + s]) : red[t] + red[t + s];
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint i = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint d = pc.headDim;
    uint kh = h / (pc.hq / pc.hkv);
    uint qBase = ((b * pc.hq + h) * pc.sq + i) * d;
    uint kvBase = (b * pc.hkv + kh) * pc.kvCapacity * d;

    for (uint c = t; c < d; c += TILE) qs[c] = q_data[qBase + c];
    barrier();

    float m = pc.hasSink != 0u ? sink_data[h] : NEG_INF;
    float l = pc.hasSink != 0u ? 1.0 : 0.0;
    float acc[COLS];
    for (uint r = 0u; r < COLS; r++) acc[r] = 0.0;

    for (uint base = 0u; base < pc.skv; base += TILE) {
        uint j = base + t;
        float s = NEG_INF;
        if (j < pc.skv) {
            uint kRow = kvBase + j * d;
            float dot = 0.0;
            for (uint c = 0u; c < d; c++) dot += qs[c] * k_data[kRow + c];
            s = pc.scale * dot;
            if (pc.softcap > 0.0) s = pc.softcap * tanh(s / pc.softcap);
        }
        float mNew = max(m, reduce(s, true));
        float corr = m == NEG_INF ? 0.0 : exp(m - mNew);
        float p = j < pc.skv && s != NEG_INF ? exp(s - mNew) : 0.0;
        ps[t] = p;
        l = l * corr + reduce(p, false);
        m = mNew;
        uint count = min(uint(TILE), pc.skv - base);
        for (uint r = 0u; r < COLS; r++) {
            uint c = t + r * TILE;
            if (c >= d) break;
            float a = acc[r] * corr;
            for (uint k = 0u; k < count; k++) a += ps[k] * v_data[kvBase + (base + k) * d + c];
            acc[r] = a;
        }
        barrier();
    }

    for (uint r = 0u; r < COLS; r++) {
        uint c = t + r * TILE;
        if (c < d) o_data[qBase + c] = acc[r] / l;
    }
}
