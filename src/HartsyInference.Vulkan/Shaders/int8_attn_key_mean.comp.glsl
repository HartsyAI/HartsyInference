// int8_attn_key_mean: the channel mean of each head's keys, as Int8AttentionV2Reference.SegmentMean takes it for one
// segment covering every valid key, so the centered keys and their bytes can match the reference bit for bit.
// Keys are F32 [heads, keys, K]; the mean is [heads, K]. One invocation per (head, channel).
//
// The order of the additions is the reference's and must stay so: partial sums over runs of 64 keys in key order, the
// partial sums added in order, then one divide by the key count. The adds are declared precise, and the divide goes
// through divRn (div_rn.glsl), because a device's own divide need not be correctly rounded.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer K_ { float k[]; };
layout(set = 0, binding = 1) writeonly buffer M_ { float mean[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint keys;
    uint K;
} pc;

const uint MEAN_CHUNK = 64u;

#include "div_rn.glsl"

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.heads * pc.K) return;
    uint head = i / pc.K, c = i - head * pc.K;
    uint base = head * pc.keys * pc.K + c;

    precise float total = 0.0;
    for (uint chunk = 0u; chunk < pc.keys; chunk += MEAN_CHUNK) {
        precise float partial = 0.0;
        uint end = min(chunk + MEAN_CHUNK, pc.keys);
        for (uint j = chunk; j < end; j++) partial = partial + k[base + j * pc.K];
        total = total + partial;
    }
    mean[i] = pc.keys > 0u ? divRn(total, float(pc.keys)) : 0.0;
}
