// norm_modulate_channel: one invocation per row along the channel axis of [N, C, ...] for IBackend.NormModulate: the C
// elements of position (n, s) sit spatial apart; y = norm(x) · w + b with KIND as norm_modulate_rows (1 LayerNorm, 2 RMSNorm,
// 3 L2-floored RMS). F32; the reduction accumulates in F32 (shaderFloat64 is not device-wide), which the contract's tolerance
// covers. Matches NormModulateReference up to that accumulation.
//
// Bindings: 0=x, 1=weight [C], 2=bias [C], 3=y
#version 460

layout(local_size_x_id = 0) in;
layout(constant_id = 3) const uint KIND = 1u;
layout(constant_id = 5) const bool HAS_WEIGHT = false;
layout(constant_id = 6) const bool HAS_BIAS = false;

layout(set = 0, binding = 0) readonly  buffer X_ { float x[]; };
layout(set = 0, binding = 1) readonly  buffer W_ { float weight[]; };
layout(set = 0, binding = 2) readonly  buffer B_ { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer Y_ { float y[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint width;
    uint spatial;
    float eps;
} pc;

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;
    uint base = (row / pc.spatial) * pc.width * pc.spatial + row % pc.spatial;
    float sum = 0.0, sumSq = 0.0;
    for (uint d = 0u; d < pc.width; d++) {
        float v = x[base + d * pc.spatial];
        sum += v;
        sumSq += v * v;
    }
    float mean = 0.0, factor = 1.0;
    if (KIND == 1u) {
        mean = sum / float(pc.width);
        float variance = 0.0;
        for (uint d = 0u; d < pc.width; d++) {
            float diff = x[base + d * pc.spatial] - mean;
            variance += diff * diff;
        }
        factor = inversesqrt(variance / float(pc.width) + pc.eps);
    } else if (KIND == 2u) {
        factor = inversesqrt(sumSq / float(pc.width) + pc.eps);
    } else if (KIND == 3u) {
        factor = sqrt(float(pc.width)) / max(sqrt(sumSq), pc.eps);
    }
    for (uint d = 0u; d < pc.width; d++) {
        float v = x[base + d * pc.spatial];
        if (KIND == 1u) v = (v - mean) * factor;
        else if (KIND != 0u) v *= factor;
        if (HAS_WEIGHT) v *= weight[d];
        if (HAS_BIAS) v += bias[d];
        y[base + d * pc.spatial] = v;
    }
}
