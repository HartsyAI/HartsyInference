// hc_split_sinkhorn: per token, split mixes[(2+hc)*hc] into pre (sigmoid + eps), post (2*sigmoid) and a Sinkhorn-normalized
// comb[hc,hc], matching HcReference.SplitSinkhorn. One thread per token; hc is at most 8 so the 64-entry matrix stays private.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Pre_ { float pre_data[]; };
layout(set = 0, binding = 1) buffer Post_ { float post_data[]; };
layout(set = 0, binding = 2) buffer Comb_ { float comb_data[]; };
layout(set = 0, binding = 3) readonly buffer Mixes_ { float mixes_data[]; };
layout(set = 0, binding = 4) readonly buffer Scale_ { float scale_data[]; };
layout(set = 0, binding = 5) readonly buffer Bias_ { float bias_data[]; };

layout(push_constant) uniform Push {
    uint tokens;
    uint hc;
    uint iters;
    float eps;
    uint tokenBase;
} pc;

// exp accurate to about one ulp on every driver (see softplus.comp.glsl): hand range reduction plus a degree-7 series.
float accExp(float x) {
    if (x > 88.0) return uintBitsToFloat(0x7F800000u);
    if (x < -87.0) return 0.0;
    float kf = floor(x * 1.44269504089 + 0.5);
    float r = fma(-kf, 0.693145751953125, x);
    r = fma(-kf, 1.42860677e-6, r);
    float p = 1.0 + r * (1.0 + r * 0.5 * (1.0 + r * (1.0 / 3.0) * (1.0 + r * 0.25 * (1.0 + r * 0.2
        * (1.0 + r * (1.0 / 6.0) * (1.0 + r * (1.0 / 7.0)))))));
    return ldexp(p, int(kf));
}

float sigmoid(float v) {
    return 1.0 / (1.0 + accExp(-v));
}

void main() {
    uint t = pc.tokenBase + gl_GlobalInvocationID.x;
    if (t >= pc.tokens) return;
    uint hc = pc.hc;
    uint width = (2u + hc) * hc;
    uint mb = t * width;
    for (uint i = 0u; i < hc; i++) {
        pre_data[t * hc + i] = sigmoid(mixes_data[mb + i] * scale_data[0] + bias_data[i]) + pc.eps;
        post_data[t * hc + i] = 2.0 * sigmoid(mixes_data[mb + hc + i] * scale_data[1] + bias_data[hc + i]);
    }
    float c[64];
    for (uint i = 0u; i < hc * hc; i++) {
        c[i] = mixes_data[mb + 2u * hc + i] * scale_data[2] + bias_data[2u * hc + i];
    }
    for (uint r = 0u; r < hc; r++) {
        float mx = c[r * hc];
        for (uint k = 1u; k < hc; k++) mx = max(mx, c[r * hc + k]);
        float sum = 0.0;
        for (uint k = 0u; k < hc; k++) {
            c[r * hc + k] = accExp(c[r * hc + k] - mx);
            sum += c[r * hc + k];
        }
        for (uint k = 0u; k < hc; k++) c[r * hc + k] = c[r * hc + k] / sum + pc.eps;
    }
    for (uint it = 0u; it < pc.iters; it++) {
        if (it > 0u) {
            for (uint r = 0u; r < hc; r++) {
                float sum = 0.0;
                for (uint k = 0u; k < hc; k++) sum += c[r * hc + k];
                for (uint k = 0u; k < hc; k++) c[r * hc + k] /= sum + pc.eps;
            }
        }
        for (uint k = 0u; k < hc; k++) {
            float sum = 0.0;
            for (uint r = 0u; r < hc; r++) sum += c[r * hc + k];
            for (uint r = 0u; r < hc; r++) c[r * hc + k] /= sum + pc.eps;
        }
    }
    for (uint i = 0u; i < hc * hc; i++) comb_data[t * hc * hc + i] = c[i];
}
