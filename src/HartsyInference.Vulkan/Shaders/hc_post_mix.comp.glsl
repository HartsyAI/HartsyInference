// hc_post_mix: output[t, i, d] = post[t, i] * x[t, d] + sum_j comb[t, j, i] * residual[t, j, d], j ascending, matching
// HcReference.PostMix. The output must not alias the residual. mix and the final sum are precise (no fused multiply-add).

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Out_ { float out_data[]; };
layout(set = 0, binding = 1) readonly buffer X_ { float x_data[]; };
layout(set = 0, binding = 2) readonly buffer Res_ { float res_data[]; };
layout(set = 0, binding = 3) readonly buffer Post_ { float post_data[]; };
layout(set = 0, binding = 4) readonly buffer Comb_ { float comb_data[]; };

layout(push_constant) uniform Push {
    uint total;
    uint hc;
    uint dim;
    uint elemBase;
} pc;

void main() {
    uint gid = pc.elemBase + gl_GlobalInvocationID.x;
    if (gid >= pc.total) return;
    uint td = pc.hc * pc.dim;
    uint t = gid / td;
    uint rem = gid - t * td;
    uint i = rem / pc.dim;
    uint d = rem - i * pc.dim;
    precise float mixv = 0.0;
    for (uint j = 0u; j < pc.hc; j++) {
        mixv += comb_data[(t * pc.hc + j) * pc.hc + i] * res_data[(t * pc.hc + j) * pc.dim + d];
    }
    precise float scaled = post_data[t * pc.hc + i] * x_data[t * pc.dim + d];
    out_data[gid] = scaled + mixv;
}
