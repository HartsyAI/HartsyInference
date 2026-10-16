// ssm_delta_prefill: the Gated-DeltaNet delta-rule recurrence over every token, one workgroup per
// (state row, value head). The rows of a head's [sv, sk] state never interact, so each row is its own scan:
// per token, row *= exp(softplus(alpha + dtBias) * ssmA); d = row . k; row += k * ((v[row] - d) * sigmoid(betaRaw));
// o[row] = row . q. Each invocation holds up to MAX_COLS of the row's sk columns. Writes the pre-norm readout; the
// gated norm runs next (ssm_gated_norm_prefill). Matches GatedDeltaNetReference.DeltaScan.
//
// Bindings: 0=output [seq, hv*sv], 1=state [hv, sv, sk] (in place), 2=q, 3=k [seq, (hv/repeat)*sk],
//           4=v [seq, hv*sv], 5=alphaRaw, 6=betaRaw [seq, hv], 7=dtBias, 8=ssmA [hv]
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
layout(set = 0, binding = 5) readonly buffer Al_ { float alphaRaw[]; };
layout(set = 0, binding = 6) readonly buffer Be_ { float betaRaw[]; };
layout(set = 0, binding = 7) readonly buffer Dt_ { float dtBias[]; };
layout(set = 0, binding = 8) readonly buffer Sa_ { float ssmA[]; };

layout(push_constant) uniform Push {
    uint seq;
    uint hv;
    uint sv;
    uint sk;
    uint repeatK;
} pc;

shared float partials[2][64];

// Workgroup-wide sum of v. The two sums of a token use different scratch rows, so each needs only its own barrier:
// every invocation reads row 0 before the barrier inside the row-1 sum, so the next token's row-0 writes, which
// come after that barrier, cannot overtake a slow reader; row 1 is covered the same way by the next row-0 barrier.
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
    uint stBase = (h * pc.sv + row) * pc.sk;
    uint lane = gl_LocalInvocationIndex;
    uint width = gl_WorkGroupSize.x;

    float s[MAX_COLS];
    for (uint i = 0u; i < MAX_COLS; i++) {
        uint c = lane + i * width;
        s[i] = c < pc.sk ? state[stBase + c] : 0.0;
    }
    float bias = dtBias[h];
    float decay = ssmA[h];
    for (uint t = 0u; t < pc.seq; t++) {
        float av = alphaRaw[t * pc.hv + h] + bias;
        float softplus = av > 20.0 ? av : log(1.0 + exp(av));
        float gExp = exp(softplus * decay);
        float beta = 1.0 / (1.0 + exp(-betaRaw[t * pc.hv + h]));
        uint kBase = t * keyDim + kh * pc.sk;
        float kr[MAX_COLS];
        float dot = 0.0;
        for (uint i = 0u; i < MAX_COLS; i++) {
            uint c = lane + i * width;
            kr[i] = c < pc.sk ? k[kBase + c] : 0.0;
            s[i] *= gExp;
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
