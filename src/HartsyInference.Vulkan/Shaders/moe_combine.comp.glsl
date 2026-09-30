// moe_combine: output[t, c] = (accumulate ? output[t, c] : 0) + sum_j w[t, j] * expertOut[pairSlot[t, j], c],
// j ascending, matching MoeReference.Combine. One thread per output element. A negative slot contributes nothing,
// and so does one past the expert rows (the reference throws there; a shader has no way to). acc is precise so the
// multiply-add is not fused: the sum has to round the way the reference's does.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Out_ { float out_data[]; };
layout(set = 0, binding = 1) readonly buffer Expert_ { float expert_data[]; };
layout(set = 0, binding = 2) readonly buffer Slots_ { int slot_data[]; };
layout(set = 0, binding = 3) readonly buffer Weights_ { float weight_data[]; };

layout(push_constant) uniform Push {
    uint tokens;
    uint width;
    uint k;
    uint expertRows;
    uint accumulate;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = pc.tokens * pc.width;
    if (gid >= total) return;
    uint t = gid / pc.width;
    uint c = gid - t * pc.width;
    precise float acc = 0.0;
    for (uint j = 0u; j < pc.k; j++) {
        int slot = slot_data[t * pc.k + j];
        if (slot < 0 || uint(slot) >= pc.expertRows) continue;
        acc += weight_data[t * pc.k + j] * expert_data[uint(slot) * pc.width + c];
    }
    out_data[gid] = (pc.accumulate != 0u) ? out_data[gid] + acc : acc;
}
