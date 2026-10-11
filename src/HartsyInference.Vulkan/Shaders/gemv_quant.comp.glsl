// Fused dequantizing GEMV for the decode-step Linear: out[m, n] = sum_k x[m, k] * W[n, k] (+ bias[n]) for M <= 8 activation
// rows, reading the GGUF weight blocks directly instead of a dequantized copy. Built three times by build.sh, one module
// per QTYPE: 0 = Q8_0 (gemv_quant_q8_0), 1 = Q4_K (gemv_quant_q4_k), 2 = Q6_K (gemv_quant_q6_k). Block layouts and the
// reconstruction formulas are the ones the dequant_q8_0 / dequant_q4_k / dequant_q6_k shaders document.
//
// One workgroup per output row. The row is cut into work units of a few words of quants (Q8_0: 4 values, 8 units per
// block; Q4_K: 16 bytes of nibbles = 32 values, 8 units per super-block; Q6_K: 4 l-positions of one half = 16 values, 16
// units per super-block) and lane t takes units t, t + WG, ..., so a subgroup reads one contiguous run of weight bytes and
// one contiguous run of activations (vec4 loads) per step. Giving each lane a whole block instead strides the activation
// loads by a block per lane and measured at about 50 GB/s on an RTX 3060.
// Plain float math: each unit dequantizes in registers and accumulates its dot per activation row; the row total is a
// subgroup sum followed by a shared-memory sum across subgroups. The integer dot-product variant (dotPacked4x8 against an
// int8 activation, as the CUDA dp4a kernels do) is a follow-up; the distro glslang cannot compile it.
//
// Q8_0 (34 bytes) and Q6_K (210 bytes) blocks are not word aligned, so those reads assemble a word from the two covering
// words; a Q4_K row is a whole number of 36-word blocks and reads plain words.

#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0) in;

layout(set = 0, binding = 0) readonly  buffer W_    { uint w_data[]; };
layout(set = 0, binding = 1) readonly  buffer X_    { vec4 x4[]; };
layout(set = 0, binding = 2) writeonly buffer Out_  { float out_data[]; };
layout(set = 0, binding = 3) readonly  buffer Bias_ { float bias_data[]; };

layout(constant_id = 1) const bool HAS_BIAS = false;
// Activation rows. A specialization constant, not a push constant, so every per-row loop has a fixed trip count and the
// accumulators stay in registers; indexed by a runtime bound they spill to local memory.
layout(constant_id = 2) const uint M = 1u;

// N output rows, K inputs per row (a multiple of the block).
layout(push_constant) uniform Push { uint N; uint K; } pc;

shared float partials[M * 32u];

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

// The four signed bytes of a word, and the low / high nibbles of its four bytes, as float vectors.
vec4 sbytes4(uint w) { return vec4(sbyte(w, 0u), sbyte(w, 1u), sbyte(w, 2u), sbyte(w, 3u)); }
vec4 lo4(uint w) { return vec4(w & 0xFu, (w >> 8) & 0xFu, (w >> 16) & 0xFu, (w >> 24) & 0xFu); }
vec4 hi4(uint w) { return vec4((w >> 4) & 0xFu, (w >> 12) & 0xFu, (w >> 20) & 0xFu, (w >> 28) & 0xFu); }

#if QTYPE == 0
#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 34u
#define UNITS_PER_BLOCK 8u
#elif QTYPE == 1
#define BLOCK_ELEMS 256u
#define BLOCK_BYTES 144u
#define Q4K_WORDS 2u
#define UNITS_PER_BLOCK (32u / Q4K_WORDS)
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
    uint K4 = pc.K >> 2;

    float acc[M];
    for (uint m = 0u; m < M; m++) acc[m] = 0.0;

    for (uint u = lane; u < units; u += wg) {
        uint b = u / UNITS_PER_BLOCK;
        uint sub = u % UNITS_PER_BLOCK;
        uint bb = rowBase + b * BLOCK_BYTES;
#if QTYPE == 0
        float d = halfAt(bb);
        vec4 q = sbytes4(readU32(bb + 2u + 4u * sub));
        uint x0 = (b * 32u + sub * 4u) >> 2;
        for (uint m = 0u; m < M; m++) acc[m] += d * dot(q, x4[m * K4 + x0]);
#elif QTYPE == 1
        // Unit sub covers qs words Q4K_WORDS*sub ..: sub-block pair p (low nibbles are sub-block 2p, high 2p+1), each word
        // four positions of both.
        uint wb = bb >> 2;
        vec2 dm = unpackHalf2x16(w_data[wb]);
        uint s0 = w_data[wb + 1u];
        uint s1 = w_data[wb + 2u];
        uint s2 = w_data[wb + 3u];
        uint p = sub / (8u / Q4K_WORDS);
        uint jl = 2u * p, jh = jl + 1u;
        uint scl, mnl, sch, mnh;
        if (p < 2u) {
            scl = ubyte(s0, jl) & 63u; mnl = ubyte(s1, jl) & 63u;
            sch = ubyte(s0, jh) & 63u; mnh = ubyte(s1, jh) & 63u;
        } else {
            scl = (ubyte(s2, jl - 4u) & 0xFu) | ((ubyte(s0, jl - 4u) >> 6) << 4);
            mnl = (ubyte(s2, jl - 4u) >> 4)   | ((ubyte(s1, jl - 4u) >> 6) << 4);
            sch = (ubyte(s2, jh - 4u) & 0xFu) | ((ubyte(s0, jh - 4u) >> 6) << 4);
            mnh = (ubyte(s2, jh - 4u) >> 4)   | ((ubyte(s1, jh - 4u) >> 6) << 4);
        }
        float dl = dm.x * float(scl), ml = dm.y * float(mnl);
        float dh = dm.x * float(sch), mh = dm.y * float(mnh);
        float accl[M], acch[M], suml[M], sumh[M];
        for (uint m = 0u; m < M; m++) { accl[m] = 0.0; acch[m] = 0.0; suml[m] = 0.0; sumh[m] = 0.0; }
        for (uint w = 0u; w < Q4K_WORDS; w++) {
            uint qw = w_data[wb + 4u + Q4K_WORDS * sub + w];
            vec4 ql = lo4(qw), qh = hi4(qw);
            uint xl = (b * 256u + p * 64u + (sub % (8u / Q4K_WORDS)) * 4u * Q4K_WORDS) / 4u + w;
            for (uint m = 0u; m < M; m++) {
                vec4 a = x4[m * K4 + xl];
                vec4 c = x4[m * K4 + xl + 8u];
                accl[m] += dot(ql, a); suml[m] += a.x + a.y + a.z + a.w;
                acch[m] += dot(qh, c); sumh[m] += c.x + c.y + c.z + c.w;
            }
        }
        for (uint m = 0u; m < M; m++) acc[m] += dl * accl[m] - ml * suml[m] + dh * acch[m] - mh * sumh[m];
#else
        // sub in [0, 16): half h = sub / 8, positions l0 .. l0 + 3 with l0 = (sub % 8) * 4; each position yields the
        // elements l, l + 32, l + 64 and l + 96 of the half.
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
        vec4 h0 = vec4(qh & 3u, (qh >> 8) & 3u, (qh >> 16) & 3u, (qh >> 24) & 3u) * 16.0;
        vec4 h1 = vec4((qh >> 2) & 3u, (qh >> 10) & 3u, (qh >> 18) & 3u, (qh >> 26) & 3u) * 16.0;
        vec4 h2 = vec4((qh >> 4) & 3u, (qh >> 12) & 3u, (qh >> 20) & 3u, (qh >> 28) & 3u) * 16.0;
        vec4 h3 = vec4((qh >> 6) & 3u, (qh >> 14) & 3u, (qh >> 22) & 3u, (qh >> 30) & 3u) * 16.0;
        vec4 q0 = lo4(qlA) + h0 - 32.0;
        vec4 q1 = lo4(qlB) + h1 - 32.0;
        vec4 q2 = hi4(qlA) + h2 - 32.0;
        vec4 q3 = hi4(qlB) + h3 - 32.0;
        uint x0 = (b * 256u + h * 128u + l0) >> 2;
        for (uint m = 0u; m < M; m++) {
            uint xb = m * K4 + x0;
            acc[m] += d0 * dot(q0, x4[xb]) + d2 * dot(q1, x4[xb + 8u]) + d4 * dot(q2, x4[xb + 16u]) + d6 * dot(q3, x4[xb + 24u]);
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
