// int8_attn_value_scales: the V scales of INT8 attention with u8 probabilities, as
// Int8AttentionV2Reference.QuantizeValues computes them. Values are F32 [heads, keys, K]. For each head, each block of
// blockKeys keys and each channel: scale = max|v| / range over the block's keys, replaced by 1 when it is below the
// smallest normal float, and NaN when any of those values is a NaN. It also writes 1 / scale, so the quantize pass
// multiplies as the reference does. Both outputs are [heads, blocks, K]. One invocation per (head, block, channel).
//
// The two divides go through divRn (div_rn.glsl): a device's own divide need not be correctly rounded.

#version 460
#extension GL_GOOGLE_include_directive : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer V_ { float v[]; };
layout(set = 0, binding = 1) writeonly buffer S_ { float scale[]; };
layout(set = 0, binding = 2) writeonly buffer I_ { float inverse[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint keys;
    uint K;
    uint blockKeys;     // keys per block; the last block may be short
    uint blocks;
    float range;        // 127
    uint scaleBase;     // the first (head, block, channel) of this dispatch
} pc;

const float MIN_NORMAL = 1.17549435e-38;

#include "div_rn.glsl"

void main() {
    uint i = pc.scaleBase + gl_GlobalInvocationID.x;
    if (i >= pc.heads * pc.blocks * pc.K) return;
    uint c = i % pc.K, hb = i / pc.K, block = hb % pc.blocks, head = hb / pc.blocks;
    uint first = block * pc.blockKeys, last = min(first + pc.blockKeys, pc.keys);
    uint base = head * pc.keys * pc.K + c;

    float maxv = 0.0;
    bool nan = false;
    for (uint j = first; j < last; j++) {
        float x = v[base + j * pc.K];
        if (isnan(x)) nan = true;
        else maxv = max(maxv, abs(x));
    }
    if (nan) {
        float q = uintBitsToFloat(0x7FC00000u);
        scale[i] = q;
        inverse[i] = q;
        return;
    }
    float sc = divRn(maxv, pc.range);
    sc = sc >= MIN_NORMAL ? sc : 1.0;
    scale[i] = sc;
    inverse[i] = divRn(1.0, sc);
}
