// Weighted sum of each token's expert-major rows through the pair map of moe_build_dispatch, plus the optional shared-expert row:
//   out[t, i] = sharedGate(t) * shared[t, i] + sum_j topkWeight[t, j] * expertOut[pairSlot[t * topk + j], i]
// sharedGate is sigmoid(sharedGateLogit[t]) when the logits are given, else 1. The sum runs in slot order starting from the shared
// term (the order of moe_combine_slots), so a dispatch is deterministic. A negative pair slot, or one past the expert rows, adds nothing.
// Unused optional bindings point at the output buffer; the shader reads them only behind the push flags.

#version 460

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) writeonly buffer Out_    { float out_data[]; };
layout(set = 0, binding = 1) readonly  buffer Expert_ { float expert_data[]; };
layout(set = 0, binding = 2) readonly  buffer Slots_  { int pair_slot[]; };
layout(set = 0, binding = 3) readonly  buffer Weight_ { float weight_data[]; };
layout(set = 0, binding = 4) readonly  buffer Shared_ { float shared_data[]; };
layout(set = 0, binding = 5) readonly  buffer Gate_   { float gate_logit[]; };

layout(push_constant) uniform Push {
    uint total;       // tokens * hidden
    uint hidden;
    uint topk;
    uint expertRows;  // rows of expertOut
    uint hasShared;
    uint hasGate;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.total) return;
    uint t = idx / pc.hidden;
    uint i = idx - t * pc.hidden;
    float acc = 0.0;
    if (pc.hasShared != 0u) {
        float s = shared_data[idx];
        if (pc.hasGate != 0u) s *= 1.0 / (1.0 + exp(-gate_logit[t]));
        acc = s;
    }
    for (uint j = 0u; j < pc.topk; j++) {
        int slot = pair_slot[t * pc.topk + j];
        if (slot < 0 || uint(slot) >= pc.expertRows) continue;
        acc += weight_data[t * pc.topk + j] * expert_data[uint(slot) * pc.hidden + i];
    }
    out_data[idx] = acc;
}
