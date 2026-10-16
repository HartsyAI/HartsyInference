// ssm_conv_split_prefill: Gated-DeltaNet prefill, the causal depthwise conv over [history ; qkvMixed] + SiLU and
// the [q | k | v] split, one invocation per (token, channel). The per-head L2 norms run next (ssm_l2norm_rows) and
// the carry update after that (ssm_conv_history_update), since this pass reads the old carry.
// Matches GatedDeltaNetReference.ConvSplit.
//
// Bindings: 0=q [seq, keyDim], 1=k [seq, keyDim], 2=v [seq, valueDim], 3=qkvMixed [seq, convDim],
//           4=history [kernel-1, convDim], 5=convW [convDim, kernel]
#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) writeonly buffer Q_ { float qOut[]; };
layout(set = 0, binding = 1) writeonly buffer K_ { float kOut[]; };
layout(set = 0, binding = 2) writeonly buffer V_ { float vOut[]; };
layout(set = 0, binding = 3) readonly  buffer M_ { float mixed[]; };
layout(set = 0, binding = 4) readonly  buffer H_ { float history[]; };
layout(set = 0, binding = 5) readonly  buffer W_ { float convW[]; };

layout(push_constant) uniform Push {
    uint convDim;
    uint keyDim;
    uint valueDim;
    uint kernelSize;
    uint seq;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.seq * pc.convDim) return;
    uint s = idx / pc.convDim;
    uint c = idx - s * pc.convDim;
    float acc = 0.0;
    for (uint j = 0u; j < pc.kernelSize; j++) {
        uint combined = s + j;
        float src = combined < pc.kernelSize - 1u ? history[combined * pc.convDim + c]
                                                  : mixed[(combined - (pc.kernelSize - 1u)) * pc.convDim + c];
        acc += convW[c * pc.kernelSize + j] * src;
    }
    float act = acc / (1.0 + exp(-acc));
    if (c < pc.keyDim) qOut[s * pc.keyDim + c] = act;
    else if (c < 2u * pc.keyDim) kOut[s * pc.keyDim + (c - pc.keyDim)] = act;
    else vOut[s * pc.valueDim + (c - 2u * pc.keyDim)] = act;
}
