// hc_pre_mix: output[t, d] = sum_i pre[t, i] * x[t, i, d], streams summed in order, matching HcReference.PreMix.
// acc is precise so the multiply-add is not fused and the sum rounds the way the reference's does.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Out_ { float out_data[]; };
layout(set = 0, binding = 1) readonly buffer X_ { float x_data[]; };
layout(set = 0, binding = 2) readonly buffer Pre_ { float pre_data[]; };

layout(push_constant) uniform Push {
    uint total;
    uint hc;
    uint dim;
    uint elemBase;
} pc;

void main() {
    uint gid = pc.elemBase + gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint t = gid / pc.dim;
    uint d = gid - t * pc.dim;
    precise float acc = 0.0;
    for (uint i = 0u; i < pc.hc; i++) {
        acc += pre_data[t * pc.hc + i] * x_data[(t * pc.hc + i) * pc.dim + d];
    }
    out_data[gid] = acc;
}
