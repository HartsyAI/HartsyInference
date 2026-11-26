// ssm_selective_scan: the Mamba-1 selective scan (IBackend.SelectiveScan), F32. Matches SelectiveScanReference.Apply.
// One workgroup of 32 invocations per (row, channel). Invocation l holds the states n = l + 32*i for i < MAX_PER_LANE
// (so up to 256 states) in registers with their decay rates, and walks the frames in order. Per frame the workgroup
// sums c·h, lane 0 writes the output. A carried state is read as the initial state and written back after the last frame;
// hasState == 0 starts at zero and writes nothing back.
//
// Bindings: 0=output [rows, frames, channels] (distinct from u), 1=u, 2=delta, 3=a [channels, states], 4=b [rows, frames, states],
//           5=c [rows, frames, states], 6=d [channels], 7=z (read only when hasZ), 8=state [rows, channels, states] (in place)
#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define MAX_PER_LANE 8
#define WG 32

layout(local_size_x = WG) in;

layout(set = 0, binding = 0) writeonly buffer O_  { float outp[]; };
layout(set = 0, binding = 1) readonly buffer U_   { float u[]; };
layout(set = 0, binding = 2) readonly buffer D_   { float delta[]; };
layout(set = 0, binding = 3) readonly buffer A_   { float a[]; };
layout(set = 0, binding = 4) readonly buffer B_   { float bm[]; };
layout(set = 0, binding = 5) readonly buffer C_   { float cm[]; };
layout(set = 0, binding = 6) readonly buffer Dd_  { float skipGain[]; };
layout(set = 0, binding = 7) readonly buffer Z_   { float z[]; };
layout(set = 0, binding = 8) buffer St_          { float state[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint frames;
    uint channels;
    uint states;
    uint hasZ;
    uint hasState;
} pc;

shared float partials[WG];

// Workgroup-wide sum; the trailing barrier keeps the next call's writes behind this call's reads.
float groupSum(float v) {
    v = subgroupAdd(v);
    if (subgroupElect()) partials[gl_SubgroupID] = v;
    barrier();
    float total = 0.0;
    for (uint i = 0u; i < gl_NumSubgroups; i++) total += partials[i];
    barrier();
    return total;
}

void main() {
    uint ch = gl_WorkGroupID.x;
    uint row = gl_WorkGroupID.y;
    if (ch >= pc.channels || row >= pc.rows) return;
    uint lane = gl_LocalInvocationIndex;
    uint S = pc.states;
    uint carriedBase = (row * pc.channels + ch) * S;

    float h[MAX_PER_LANE];
    float r[MAX_PER_LANE];
    for (uint i = 0u; i < MAX_PER_LANE; i++) {
        uint n = lane + WG * i;
        bool live = n < S;
        h[i] = (live && pc.hasState != 0u) ? state[carriedBase + n] : 0.0;
        r[i] = live ? a[ch * S + n] : 0.0;
    }
    float skip = skipGain[ch];

    for (uint t = 0u; t < pc.frames; t++) {
        uint at = (row * pc.frames + t) * pc.channels + ch;
        uint bBase = (row * pc.frames + t) * S;
        float step = delta[at];
        float x = u[at];
        float drive = step * x;
        float acc = 0.0;
        for (uint i = 0u; i < MAX_PER_LANE; i++) {
            uint n = lane + WG * i;
            if (n < S) {
                h[i] = exp(step * r[i]) * h[i] + drive * bm[bBase + n];
                acc += h[i] * cm[bBase + n];
            }
        }
        acc = groupSum(acc);
        if (lane == 0u) {
            float y = acc + x * skip;
            if (pc.hasZ != 0u) {
                float g = z[at];
                y *= g / (1.0 + exp(-g));
            }
            outp[at] = y;
        }
    }

    if (pc.hasState != 0u) {
        for (uint i = 0u; i < MAX_PER_LANE; i++) {
            uint n = lane + WG * i;
            if (n < S) state[carriedBase + n] = h[i];
        }
    }
}
