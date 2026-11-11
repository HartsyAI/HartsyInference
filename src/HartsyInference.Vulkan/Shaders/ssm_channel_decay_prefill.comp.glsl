// ssm_channel_decay_prefill: the delta-rule recurrence with a per-key-channel log-decay (Kimi KDA), over every token, one
// workgroup per (state row, value head). Same layout and token order as ssm_delta_prefill; the difference is the decay:
// column c of the row decays by exp(logDecay[t, h, c]) instead of one scalar per head. Per token:
// row[c] *= exp(logDecay[c]); d = row . k; row += k * ((v[row] - d) * sigmoid(betaRaw)); o[row] = row . q.
// Writes the raw readout (no norm, no gate). Matches GatedDeltaNetReference.ChannelDecayScan.
//
// Bindings: 0=output [seq, hv*sv], 1=state [hv, sv, sk] (in place), 2=q, 3=k [seq, (hv/repeat)*sk],
//           4=v [seq, hv*sv], 5=logDecay [seq, hv*sk], 6=betaRaw [seq, hv]
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define MAX_COLS 4

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) writeonly buffer O_  { float outp[]; };
layout(set = 0, binding = 1) buffer St_ { float state[]; };
layout(set = 0, binding = 2) readonly buffer Q_  { float q[]; };
layout(set = 0, binding = 3) readonly buffer K_  { float k[]; };
layout(set = 0, binding = 4) readonly buffer V_  { float v[]; };
layout(set = 0, binding = 5) readonly buffer G_  { float logDecay[]; };
layout(set = 0, binding = 6) readonly buffer Be_ { float betaRaw[]; };

layout(push_constant) uniform Push {
    uint seq;
    uint hv;
    uint sv;
    uint sk;
    uint repeatK;
} pc;

shared float partials[2][64];

// Workgroup-wide sum of v, with the same two-slot scratch scheme as ssm_delta_prefill.
float groupSum(float v, uint slot) {
    v = subgroupAdd(v);
    if (subgroupElect()) partials[slot][gl_SubgroupID] = v;
    barrier();
    float total = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; i++) total += partials[slot][i];
    return total;
}

void main() {
    uint row = gl_WorkGroupID.x;
    uint h = gl_WorkGroupID.y;
    if (row >= pc.sv || h >= pc.hv) return;
    uint kh = h / pc.repeatK;
    uint keyDim = (pc.hv / pc.repeatK) * pc.sk;
    uint valueDim = pc.hv * pc.sv;
    uint decayDim = pc.hv * pc.sk;
    uint stBase = (h * pc.sv + row) * pc.sk;
    uint lane = gl_LocalInvocationIndex;
    uint width = gl_WorkGroupSize.x;

    float s[MAX_COLS];
    for (uint i = 0u; i < MAX_COLS; i++) {
        uint c = lane + i * width;
        s[i] = c < pc.sk ? state[stBase + c] : 0.0;
    }
    for (uint t = 0u; t < pc.seq; t++) {
        float beta = 1.0 / (1.0 + exp(-betaRaw[t * pc.hv + h]));
        uint kBase = t * keyDim + kh * pc.sk;
        uint gBase = t * decayDim + h * pc.sk;
        float kr[MAX_COLS];
        float dot = 0.0;
        for (uint i = 0u; i < MAX_COLS; i++) {
            uint c = lane + i * width;
            kr[i] = c < pc.sk ? k[kBase + c] : 0.0;
            if (c < pc.sk) s[i] *= exp(logDecay[gBase + c]);
            dot += s[i] * kr[i];
        }
        dot = groupSum(dot, 0u);
        float delta = (v[t * valueDim + h * pc.sv + row] - dot) * beta;
        float o = 0.0;
        for (uint i = 0u; i < MAX_COLS; i++) {
            uint c = lane + i * width;
            s[i] += kr[i] * delta;
            o += c < pc.sk ? s[i] * q[kBase + c] : 0.0;
        }
        o = groupSum(o, 1u);
        if (lane == 0u) outp[t * valueDim + h * pc.sv + row] = o;
    }
    for (uint i = 0u; i < MAX_COLS; i++) {
        uint c = lane + i * width;
        if (c < pc.sk) state[stBase + c] = s[i];
    }
}
