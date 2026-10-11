// Fused dequantizing GEMV for the decode-step Linear: out[m, n] = sum_k x[m, k] * W[n, k] (+ bias[n]) for M <= 8 activation
// rows, reading the GGUF weight blocks directly instead of a dequantized copy. Built three times by build.sh, one module
// per QTYPE: 0 = Q8_0 (gemv_q8_0), 1 = Q4_K (gemv_q4_k), 2 = Q6_K (gemv_q6_k). Block layouts and the reconstruction
// formulas are the ones the dequant_q8_0 / dequant_q4_k / dequant_q6_k shaders document.
//
// One workgroup per output row. The row's blocks are split into fixed work units (Q8_0: one 32-element block; Q4_K: a
// pair of 32-element sub-blocks, four units per super-block; Q6_K: four of the 32 l-positions of one half, sixteen units
// per super-block), and lane t takes units t, t + WG, ... so neighbouring lanes read neighbouring bytes. Plain float math:
// each unit dequantizes in registers and accumulates its dot with the activation per column; the row total is a
// subgroup sum followed by a shared-memory sum across subgroups. The integer dot-product variant (dotPacked4x8 against an
// int8 activation, as the CUDA dp4a kernels do) is a follow-up; the distro glslang cannot compile it.
//
// Weight rows are byte runs of K / blockElems blocks; Q8_0 (34 bytes) and Q6_K (210 bytes) blocks are not word aligned,
// so unaligned 32-bit reads are assembled from the two covering words.

#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require

#define MAX_M 8u

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer W_    { uint w_data[]; };
layout(set = 0, binding = 1) readonly  buffer X_    { float x_data[]; };
layout(set = 0, binding = 2) writeonly buffer Out_  { float out_data[]; };
layout(set = 0, binding = 3) readonly  buffer Bias_ { float bias_data[]; };

layout(constant_id = 1) const bool HAS_BIAS = false;

// N output rows, K inputs per row, M activation rows (1..MAX_M).
layout(push_constant) uniform Push { uint N; uint K; uint M; } pc;

shared float partials[MAX_M * 32u];

uint readU32(uint byteOffset) {
    uint idx = byteOffset >> 2;
    uint sh = (byteOffset & 3u) * 8u;
    uint lo = w_data[idx];
    if (sh == 0u) return lo;
    return (lo >> sh) | (w_data[idx + 1u] << (32u - sh));
}

uint readU16(uint byteOffset) {
    return (w_data[byteOffset >> 2] >> ((byteOffset & 2u) * 8u)) & 0xFFFFu;
}

float halfAt(uint byteOffset) { return unpackHalf2x16(readU16(byteOffset)).x; }

int sbyte(uint word, uint i) { return int(word << (24u - 8u * i)) >> 24; }
uint ubyte(uint word, uint i) { return (word >> (8u * i)) & 0xFFu; }

#if QTYPE == 0
#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 34u
#define UNITS_PER_BLOCK 1u
#elif QTYPE == 1
#define BLOCK_ELEMS 256u
#define BLOCK_BYTES 144u
#define UNITS_PER_BLOCK 4u
#else
#define BLOCK_ELEMS 256u
#define BLOCK_BYTES 210u
#define UNITS_PER_BLOCK 16u
#endif

void main() {
    // A vocabulary-sized N exceeds the 65535 workgroups a dimension is guaranteed, so the grid may be two-dimensional.
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= pc.N) return;
    uint lane = gl_LocalInvocationID.x;
    uint wg = gl_WorkGroupSize.x;
    uint blocksPerRow = pc.K / BLOCK_ELEMS;
    uint units = blocksPerRow * UNITS_PER_BLOCK;
    uint rowBase = row * blocksPerRow * BLOCK_BYTES;
    uint M = min(pc.M, MAX_M);

    float acc[MAX_M];
    for (uint m = 0u; m < MAX_M; m++) acc[m] = 0.0;

    for (uint u = lane; u < units; u += wg) {
        uint b = u / UNITS_PER_BLOCK;
        uint sub = u % UNITS_PER_BLOCK;
        uint bb = rowBase + b * BLOCK_BYTES;
#if QTYPE == 0
        float d = halfAt(bb);
        float q[32];
        for (uint i = 0u; i < 8u; i++) {
            uint word = readU32(bb + 2u + 4u * i);
            for (uint j = 0u; j < 4u; j++) q[4u * i + j] = float(sbyte(word, j));
        }
        uint k0 = b * 32u;
        for (uint m = 0u; m < M; m++) {
            uint xb = m * pc.K + k0;
            float s = 0.0;
            for (uint i = 0u; i < 32u; i++) s += q[i] * x_data[xb + i];
            acc[m] += d * s;
        }
#elif QTYPE == 1
        // Rows of a Q4_K weight start word aligned (144 = 36 words), so every read here is a plain word.
        uint hdr = w_data[bb >> 2];
        vec2 dm = unpackHalf2x16(hdr);
        uint s0 = w_data[(bb >> 2) + 1u];
        uint s1 = w_data[(bb >> 2) + 2u];
        uint s2 = w_data[(bb >> 2) + 3u];
        // get_scale_min_k4 for j = 2*sub (low nibbles) and j + 1 (high nibbles).
        uint jl = 2u * sub, jh = jl + 1u;
        uint scl, mnl, sch, mnh;
        if (jl < 4u) {
            scl = ubyte(s0, jl) & 63u;        mnl = ubyte(s1, jl) & 63u;
            sch = ubyte(s0, jh) & 63u;        mnh = ubyte(s1, jh) & 63u;
        } else {
            scl = (ubyte(s2, jl - 4u) & 0xFu) | ((ubyte(s0, jl - 4u) >> 6) << 4);
            mnl = (ubyte(s2, jl - 4u) >> 4)   | ((ubyte(s1, jl - 4u) >> 6) << 4);
            sch = (ubyte(s2, jh - 4u) & 0xFu) | ((ubyte(s0, jh - 4u) >> 6) << 4);
            mnh = (ubyte(s2, jh - 4u) >> 4)   | ((ubyte(s1, jh - 4u) >> 6) << 4);
        }
        float dl = dm.x * float(scl), ml = dm.y * float(mnl);
        float dh = dm.x * float(sch), mh = dm.y * float(mnh);
        uint qw[8];
        uint qBase = ((bb + 16u) >> 2) + sub * 8u;
        for (uint i = 0u; i < 8u; i++) qw[i] = w_data[qBase + i];
        uint k0 = b * 256u + sub * 64u;
        for (uint m = 0u; m < M; m++) {
            uint xb = m * pc.K + k0;
            float sl = 0.0, sxl = 0.0, sh = 0.0, sxh = 0.0;
            for (uint i = 0u; i < 32u; i++) {
                uint byte_ = ubyte(qw[i >> 2], i & 3u);
                float xl = x_data[xb + i];
                float xh = x_data[xb + 32u + i];
                sl += float(byte_ & 0xFu) * xl; sxl += xl;
                sh += float(byte_ >> 4) * xh;   sxh += xh;
            }
            acc[m] += dl * sl - ml * sxl + dh * sh - mh * sxh;
        }
#else
        // sub in [0, 16): half h = sub / 8, positions l0 .. l0 + 3 with l0 = (sub % 8) * 4.
        uint h = sub >> 3;
        uint l0 = (sub & 7u) * 4u;
        uint qlA = readU32(bb + h * 64u + l0);
        uint qlB = readU32(bb + h * 64u + l0 + 32u);
        uint qh = readU32(bb + 128u + h * 32u + l0);
        uint scBase = bb + 192u + h * 8u + (l0 >> 4);
        float d = halfAt(bb + 208u);
        uint scw0 = readU32(scBase);
        uint scw1 = readU32(scBase + 4u);
        float d0 = d * float(sbyte(scw0, 0u));
        float d2 = d * float(sbyte(scw0, 2u));
        float d4 = d * float(sbyte(scw1, 0u));
        float d6 = d * float(sbyte(scw1, 2u));
        uint k0 = b * 256u + h * 128u + l0;
        for (uint m = 0u; m < M; m++) {
            uint xb = m * pc.K + k0;
            float s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (uint i = 0u; i < 4u; i++) {
                uint a = ubyte(qlA, i), c = ubyte(qlB, i), e = ubyte(qh, i);
                s0 += float(int((a & 0xFu) | ((e & 3u) << 4)) - 32) * x_data[xb + i];
                s1 += float(int((c & 0xFu) | (((e >> 2) & 3u) << 4)) - 32) * x_data[xb + 32u + i];
                s2 += float(int((a >> 4) | (((e >> 4) & 3u) << 4)) - 32) * x_data[xb + 64u + i];
                s3 += float(int((c >> 4) | (((e >> 6) & 3u) << 4)) - 32) * x_data[xb + 96u + i];
            }
            acc[m] += d0 * s0 + d2 * s1 + d4 * s2 + d6 * s3;
        }
#endif
    }

    // Row total: subgroup sum, then one partial per subgroup through shared memory.
    uint sgCount = (wg + gl_SubgroupSize - 1u) / gl_SubgroupSize;
    for (uint m = 0u; m < M; m++) {
        float s = subgroupAdd(acc[m]);
        if (subgroupElect()) partials[m * 32u + gl_SubgroupID] = s;
    }
    barrier();
    if (lane < M) {
        float total = 0.0;
        for (uint g = 0u; g < sgCount; g++) total += partials[lane * 32u + g];
        if (HAS_BIAS) total += bias_data[row];
        out_data[lane * pc.N + row] = total;
    }
}
