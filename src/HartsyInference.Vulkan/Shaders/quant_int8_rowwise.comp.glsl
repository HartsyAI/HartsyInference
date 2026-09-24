// Per-row symmetric INT8 quantization of an F32 [rows, K] matrix, the GPU form of Int8Quantizer.RowwiseSymmetric:
// scale[r] = max|x[r]| / 127 (1 for an all-zero row), q = clamp(roundEven(x / scale), -127, 127), four int8 packed
// per int word in element order (what matmul_int8 reads). The scales land at scaleOffset floats into the scale
// binding, so one buffer can carry [packed int8 | scales] and be bound as both. One workgroup per row.

#version 460
#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer X_ { float x[]; };
layout(set = 0, binding = 1) writeonly buffer Q_ { int q[]; };
layout(set = 0, binding = 2) writeonly buffer S_ { float s[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint K;             // a multiple of 4
    uint scaleOffset;   // float offset of the scale region in the scale binding
} pc;

shared float warp_max[64];   // one slot per subgroup of the 256-invocation workgroup: 64 covers any subgroup size of 4 or more (softmax.comp.glsl's idiom)
shared float gScale;

int quantOne(float v, float inv) {
    return clamp(int(roundEven(v * inv)), -127, 127);
}

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint base = row * pc.K;

    float maxv = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < pc.K; i += gl_WorkGroupSize.x)
        maxv = max(maxv, abs(x[base + i]));
    maxv = subgroupMax(maxv);
    if (subgroupElect()) warp_max[gl_SubgroupID] = maxv;
    barrier();
    if (gl_SubgroupID == 0u) {
        float v = 0.0;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize)
            v = max(v, warp_max[k]);
        v = subgroupMax(v);
        if (subgroupElect()) gScale = v > 0.0 ? v / 127.0 : 1.0;
    }
    barrier();

    float scale = gScale;
    float inv = 1.0 / scale;
    uint words = pc.K / 4u;
    for (uint w = gl_LocalInvocationIndex; w < words; w += gl_WorkGroupSize.x) {
        uint e = base + w * 4u;
        int q0 = quantOne(x[e], inv), q1 = quantOne(x[e + 1u], inv), q2 = quantOne(x[e + 2u], inv), q3 = quantOne(x[e + 3u], inv);
        q[row * words + w] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | (q3 << 24);
    }
    if (gl_LocalInvocationIndex == 0u) s[pc.scaleOffset + row] = scale;
}
