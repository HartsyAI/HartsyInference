// int8_attn_quant_rows: the row quantization of INT8 attention, written to give Int8AttentionV2Reference's bytes and
// scales bit for bit. Per row of an F32 [rows, K] matrix: scale = max|x| / range, replaced by 1 when it is below the
// smallest normal float; byte = roundEven(clamp(x * (1 / scale), -127, 127)); a row holding a NaN gets a NaN scale and
// zero bytes. Four bytes are packed per int word in element order. One workgroup per row.
//
// It differs from quant_int8_rowwise, which the INT8 linear layers use, in three ways that matter here: its two divides
// are corrected to the correctly rounded quotient; the scale has the reference's floor; and a NaN row is defined.
//
// The correction: Vulkan allows a divide to be a few units in the last place off, and on the RTX 3060 this was written
// on it is (about one scale in twenty came out one unit low, with a constant or a run-time divisor alike). divRn takes
// the device's quotient q0, forms the exact remainder a - q0 * b with one fused multiply-add, and adds remainder / b.
// The remainder is exact because q0 is within a few units of the true quotient, and a quotient of two floats is never
// exactly halfway between two floats, so the sum rounds to the correctly rounded quotient. It needs fma to be fused,
// which a precise fma is.

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
    float range;        // 127 for Q and K
} pc;

const float MIN_NORMAL = 1.17549435e-38;

shared float warp_max[64];   // one slot per subgroup of the workgroup, as in quant_int8_rowwise
shared uint warp_nan[64];
shared float gScale;
shared uint gNan;

float divRn(float a, float b) {
    precise float q0 = a / b;
    precise float remainder = fma(-q0, b, a);
    precise float correction = remainder / b;
    precise float q = q0 + correction;
    return q;
}

int quantOne(float v, float inv) {
    precise float scaled = v * inv;
    return int(roundEven(clamp(scaled, -127.0, 127.0)));
}

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint base = row * pc.K;

    float maxv = 0.0;
    uint nan = 0u;
    for (uint i = gl_LocalInvocationIndex; i < pc.K; i += gl_WorkGroupSize.x) {
        float v = x[base + i];
        if (isnan(v)) nan = 1u;
        else maxv = max(maxv, abs(v));
    }
    maxv = subgroupMax(maxv);
    nan = subgroupMax(nan);
    if (subgroupElect()) {
        warp_max[gl_SubgroupID] = maxv;
        warp_nan[gl_SubgroupID] = nan;
    }
    barrier();
    if (gl_SubgroupID == 0u) {
        float v = 0.0;
        uint n = 0u;
        for (uint k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize) {
            v = max(v, warp_max[k]);
            n = max(n, warp_nan[k]);
        }
        v = subgroupMax(v);
        n = subgroupMax(n);
        if (subgroupElect()) {
            float sc = divRn(v, pc.range);
            gScale = sc >= MIN_NORMAL ? sc : 1.0;
            gNan = n;
        }
    }
    barrier();

    bool rowNan = gNan != 0u;
    float scale = gScale;
    float inv = divRn(1.0, scale);
    uint words = pc.K / 4u;
    for (uint w = gl_LocalInvocationIndex; w < words; w += gl_WorkGroupSize.x) {
        uint e = base + w * 4u;
        int packed = 0;
        if (!rowNan) {
            int q0 = quantOne(x[e], inv), q1 = quantOne(x[e + 1u], inv), q2 = quantOne(x[e + 2u], inv), q3 = quantOne(x[e + 3u], inv);
            packed = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | (q3 << 24);
        }
        q[row * words + w] = packed;
    }
    if (gl_LocalInvocationIndex == 0u) s[row] = rowNan ? uintBitsToFloat(0x7FC00000u) : scale;
}
