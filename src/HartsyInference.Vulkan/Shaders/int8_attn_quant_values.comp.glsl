// int8_attn_quant_values: the V bytes of INT8 attention with u8 probabilities, from the reciprocal scales
// int8_attn_value_scales wrote. Values are F32 [heads, keys, K]; byte = roundEven(clamp(v * inverse, -127, 127)) with
// the inverse of the value's (head, block, channel), and 0 where that product is a NaN, as the reference's ToS8 gives.
// Four bytes are packed per int word in element order. One invocation per word.

#version 460

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer V_ { float v[]; };
layout(set = 0, binding = 1) readonly  buffer I_ { float inverse[]; };
layout(set = 0, binding = 2) writeonly buffer Q_ { int q[]; };

layout(push_constant) uniform Push {
    uint heads;
    uint keys;
    uint K;             // a multiple of 4
    uint blockKeys;
    uint blocks;
    uint wordBase;      // the first word of this dispatch
} pc;

int quantOne(float x, float inv) {
    precise float scaled = x * inv;
    if (isnan(scaled)) return 0;
    return int(roundEven(clamp(scaled, -127.0, 127.0)));
}

void main() {
    uint w = pc.wordBase + gl_GlobalInvocationID.x;
    uint wordsPerRow = pc.K / 4u;
    if (w >= pc.heads * pc.keys * wordsPerRow) return;
    uint row = w / wordsPerRow, c = (w - row * wordsPerRow) * 4u;
    uint head = row / pc.keys, key = row - head * pc.keys;
    uint e = row * pc.K + c;
    uint s = (head * pc.blocks + key / pc.blockKeys) * pc.K + c;
    int q0 = quantOne(v[e], inverse[s]), q1 = quantOne(v[e + 1u], inverse[s + 1u]);
    int q2 = quantOne(v[e + 2u], inverse[s + 2u]), q3 = quantOne(v[e + 3u], inverse[s + 3u]);
    q[w] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | (q3 << 24);
}
