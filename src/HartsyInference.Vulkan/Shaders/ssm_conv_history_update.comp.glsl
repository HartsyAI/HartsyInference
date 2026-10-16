// ssm_conv_history_update: advances the Gated-DeltaNet conv carry to this call's last (kernel-1) pre-conv rows,
// one invocation per channel. Runs after ssm_conv_split_prefill, which reads the old carry.
//
// Bindings: 0=history [kernel-1, convDim] (in place), 1=qkvMixed [seq, convDim]
#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer H_ { float history[]; };
layout(set = 0, binding = 1) readonly buffer M_ { float mixed[]; };

layout(push_constant) uniform Push {
    uint convDim;
    uint kernelSize;
    uint seq;
} pc;

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= pc.convDim || pc.kernelSize <= 1u) return;
    uint carry = min(pc.seq, pc.kernelSize - 1u);
    for (uint r = 0u; r + carry < pc.kernelSize - 1u; r++)
        history[r * pc.convDim + c] = history[(r + carry) * pc.convDim + c];
    for (uint r = 0u; r < carry; r++)
        history[(pc.kernelSize - 1u - carry + r) * pc.convDim + c] = mixed[(pc.seq - carry + r) * pc.convDim + c];
}
