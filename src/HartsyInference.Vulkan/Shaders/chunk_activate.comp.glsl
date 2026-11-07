// chunk_activate: cuts the last axis of x [rows, count·width] into count outputs [rows, width], chunk c through act[c]
// (0 identity, 1 1+x, 2 tanh, 3 sigmoid): an adaLN modulation projection split into its scales and gates in one pass
// (IBackend.ChunkActivate). The output bindings past count are placeholders and never written. F32.
//
// Bindings: 0=x, 1..8=outputs 0..7
#version 460

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_  { float x[]; };
layout(set = 0, binding = 1) writeonly buffer O0_ { float out0[]; };
layout(set = 0, binding = 2) writeonly buffer O1_ { float out1[]; };
layout(set = 0, binding = 3) writeonly buffer O2_ { float out2[]; };
layout(set = 0, binding = 4) writeonly buffer O3_ { float out3[]; };
layout(set = 0, binding = 5) writeonly buffer O4_ { float out4[]; };
layout(set = 0, binding = 6) writeonly buffer O5_ { float out5[]; };
layout(set = 0, binding = 7) writeonly buffer O6_ { float out6[]; };
layout(set = 0, binding = 8) writeonly buffer O7_ { float out7[]; };

layout(push_constant) uniform Push {
    uint total;
    uint width;
    uint count;
    uint act0; uint act1; uint act2; uint act3; uint act4; uint act5; uint act6; uint act7;
} pc;

uint act_of(uint c) {
    if (c == 0u) return pc.act0; if (c == 1u) return pc.act1; if (c == 2u) return pc.act2; if (c == 3u) return pc.act3;
    if (c == 4u) return pc.act4; if (c == 5u) return pc.act5; if (c == 6u) return pc.act6; return pc.act7;
}

float activate(float v, uint act) {
    if (act == 1u) return 1.0 + v;
    if (act == 2u) return tanh(v);
    if (act == 3u) return 1.0 / (1.0 + exp(-v));
    return v;
}

void store(uint c, uint at, float v) {
    if (c == 0u) out0[at] = v; else if (c == 1u) out1[at] = v; else if (c == 2u) out2[at] = v; else if (c == 3u) out3[at] = v;
    else if (c == 4u) out4[at] = v; else if (c == 5u) out5[at] = v; else if (c == 6u) out6[at] = v; else out7[at] = v;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint packed = pc.width * pc.count;
    uint row = i / packed, rem = i % packed;
    uint c = rem / pc.width, d = rem % pc.width;
    store(c, row * pc.width + d, activate(x[i], act_of(c)));
}
