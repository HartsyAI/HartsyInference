// Weighted sum of each token's routed-expert rows, plus the optional shared-expert row, in one dispatch:
//   out[t, i] = sharedGate(t) * shared[t, i] + sum_j topkWeight[t, j] * slotOut[t * topk + j, i]
// sharedGate is sigmoid(sharedGateLogit[t]) when the logits are given, else 1; with no shared row the first term is 0. The
// sum runs in slot order, starting from the shared term, so a dispatch is deterministic (the order of CUDA's moe_combine_slots).
// Unused optional bindings point at the output buffer; the shader reads them only behind the push flags.

#version 460

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) writeonly buffer Out_    { float out_data[]; };
layout(set = 0, binding = 1) readonly  buffer Slots_  { float slot_data[]; };
layout(set = 0, binding = 2) readonly  buffer Weight_ { float weight_data[]; };
layout(set = 0, binding = 3) readonly  buffer Shared_ { float shared_data[]; };
layout(set = 0, binding = 4) readonly  buffer Gate_   { float gate_logit[]; };

layout(push_constant) uniform Push {
    uint total;      // tokens * hidden
    uint hidden;
    uint topk;
    uint hasShared;  // 1 when binding 3 holds the shared row
    uint hasGate;    // 1 when binding 4 holds the pre-sigmoid shared-gate logit per token
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
        acc += weight_data[t * pc.topk + j] * slot_data[(t * pc.topk + j) * pc.hidden + i];
    }
    out_data[idx] = acc;
}
