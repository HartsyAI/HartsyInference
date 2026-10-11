// The per-work-unit decoder of the fused dequantizing GEMV, shared by gemv_quant (dense decode Linear) and the MoE expert
// shaders (moe_expert_gate_up, moe_expert_down), so one inner loop serves every user. A unit is a few words of quants (Q8_0:
// 4 values, 8 units per block; Q4_K: 16 bytes of nibbles = 32 values, 8 units per super-block; Q6_K: 4 l-positions of one half
// = 16 values, 16 units per super-block; F32: 4 floats) and is dequantized in registers and dotted against M activation rows.
// Block layouts and reconstruction formulas are the ones the dequant_<type> shaders document.
//
// QTYPE: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = Q4_0, 4 = Q5_0, 5 = Q5_K, 6 = F32.
//
// The includer declares, before each include, a uint-array buffer and the names this file defines through macros, so the
// file can be included once per weight buffer (the gate/up shader includes it twice):
//   WBUF       the weight buffer's array (uint[])
//   FN_READ32  name for the unaligned 32-bit read helper      FN_READ16  the 16-bit read helper
//   FN_HALF    name for the half-float read helper            FN_UNIT    name for the unit function
//   ACC        the includer's float[M] accumulator array (private global), M the activation-row count (a specialization
//              constant: indexed by a runtime bound the accumulators spill to local memory)
// and a vec4 buffer x4 holding the activation rows. FN_UNIT(u, rowBase, K4, xOff) adds unit u of the row starting at byte
// rowBase of WBUF into ACC for every activation row m, which starts xOff + m * K4 vec4s into x4.
//
// Q8_0 (34 bytes), Q4_0 (18), Q5_0 (22) and Q6_K (210) blocks are not word aligned, so those reads assemble a word from
// the two covering words; Q4_K (144), Q5_K (176) and F32 rows are whole numbers of words and read plain words.

#ifndef GEMV_UNIT_COMMON
#define GEMV_UNIT_COMMON

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
#elif QTYPE == 2
#define BLOCK_ELEMS 256u
#define BLOCK_BYTES 210u
#define UNITS_PER_BLOCK 16u
#elif QTYPE == 3
#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 18u
#define UNITS_PER_BLOCK 4u
#elif QTYPE == 4
#define BLOCK_ELEMS 32u
#define BLOCK_BYTES 22u
#define UNITS_PER_BLOCK 4u
#elif QTYPE == 6
#define BLOCK_ELEMS 4u
#define BLOCK_BYTES 16u
#define UNITS_PER_BLOCK 1u
#else
#define BLOCK_ELEMS 256u
#define BLOCK_BYTES 176u
#define Q4K_WORDS 2u
#define UNITS_PER_BLOCK (32u / Q4K_WORDS)
#endif


#endif

uint FN_READ32(uint byteOffset) {
    uint idx = byteOffset >> 2;
    uint sh = (byteOffset & 3u) * 8u;
    uint lo = WBUF[idx];
    if (sh == 0u) return lo;
    return (lo >> sh) | (WBUF[idx + 1u] << (32u - sh));
}

uint FN_READ16(uint byteOffset) {
    return (WBUF[byteOffset >> 2] >> ((byteOffset & 2u) * 8u)) & 0xFFFFu;
}

float FN_HALF(uint byteOffset) { return unpackHalf2x16(FN_READ16(byteOffset)).x; }

void FN_UNIT(uint u, uint rowBase, uint K4, uint xOff) {
    uint b = u / UNITS_PER_BLOCK;
    uint sub = u % UNITS_PER_BLOCK;
    uint bb = rowBase + b * BLOCK_BYTES;
#if QTYPE == 0
    float d = FN_HALF(bb);
    vec4 q = sbytes4(FN_READ32(bb + 2u + 4u * sub));
    uint x0 = (b * 32u + sub * 4u) >> 2;
    for (uint m = 0u; m < M; m++) ACC[m] += d * dot(q, x4[xOff + m * K4 + x0]);
#elif QTYPE == 1 || QTYPE == 5
    // Q4_K, and Q5_K with its high bits. Unit sub covers qs words Q4K_WORDS*sub ..: sub-block pair p (low nibbles are sub-block 2p, high 2p+1), each word
    // four positions of both.
    uint wb = bb >> 2;
    vec2 dm = unpackHalf2x16(WBUF[wb]);
    uint s0 = WBUF[wb + 1u];
    uint s1 = WBUF[wb + 2u];
    uint s2 = WBUF[wb + 3u];
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
#if QTYPE == 1
        uint qw = WBUF[wb + 4u + Q4K_WORDS * sub + w];
        vec4 ql = lo4(qw), qh = hi4(qw);
#else
        // Q5_K: qs follows 32 bytes of high bits; byte l of qh carries bit 2p (low nibble) and 2p+1 (high) for position l.
        uint qw = WBUF[wb + 12u + Q4K_WORDS * sub + w];
        uint hw = WBUF[wb + 4u + (sub % (8u / Q4K_WORDS)) * Q4K_WORDS + w] >> (2u * p);
        vec4 ql = lo4(qw) + vec4(hw & 1u, (hw >> 8) & 1u, (hw >> 16) & 1u, (hw >> 24) & 1u) * 16.0;
        vec4 qh = hi4(qw) + vec4((hw >> 1) & 1u, (hw >> 9) & 1u, (hw >> 17) & 1u, (hw >> 25) & 1u) * 16.0;
#endif
        uint xl = (b * 256u + p * 64u + (sub % (8u / Q4K_WORDS)) * 4u * Q4K_WORDS) / 4u + w;
        for (uint m = 0u; m < M; m++) {
            vec4 a = x4[xOff + m * K4 + xl];
            vec4 c = x4[xOff + m * K4 + xl + 8u];
            accl[m] += dot(ql, a); suml[m] += a.x + a.y + a.z + a.w;
            acch[m] += dot(qh, c); sumh[m] += c.x + c.y + c.z + c.w;
        }
    }
    for (uint m = 0u; m < M; m++) ACC[m] += dl * accl[m] - ml * suml[m] + dh * acch[m] - mh * sumh[m];
#elif QTYPE == 6
    // F32: unit b is the four floats at elements 4b .. 4b + 3 of the row.
    uint fw = bb >> 2;
    vec4 q = uintBitsToFloat(uvec4(WBUF[fw], WBUF[fw + 1u], WBUF[fw + 2u], WBUF[fw + 3u]));
    for (uint m = 0u; m < M; m++) ACC[m] += dot(q, x4[xOff + m * K4 + b]);
#elif QTYPE == 3 || QTYPE == 4
    // Q4_0 / Q5_0: unit sub covers qs bytes 4*sub .. +3, whose low nibbles are positions i .. i+3 (i = 4*sub) and high
    // nibbles positions i+16 .. i+19; Q5_0 adds bit i (and i+16) of the 32-bit qh word as the fifth bit.
    float d = FN_HALF(bb);
#if QTYPE == 3
    uint qw = FN_READ32(bb + 2u + 4u * sub);
    vec4 qa = lo4(qw) - 8.0, qb = hi4(qw) - 8.0;
#else
    uint qhw = FN_READ32(bb + 2u);
    uint qw = FN_READ32(bb + 6u + 4u * sub);
    uint ha = qhw >> (4u * sub), hb = qhw >> (4u * sub + 16u);
    vec4 qa = lo4(qw) + vec4(ha & 1u, (ha >> 1) & 1u, (ha >> 2) & 1u, (ha >> 3) & 1u) * 16.0 - 16.0;
    vec4 qb = hi4(qw) + vec4(hb & 1u, (hb >> 1) & 1u, (hb >> 2) & 1u, (hb >> 3) & 1u) * 16.0 - 16.0;
#endif
    uint x0 = (b * 32u + sub * 4u) >> 2;
    for (uint m = 0u; m < M; m++) ACC[m] += d * (dot(qa, x4[xOff + m * K4 + x0]) + dot(qb, x4[xOff + m * K4 + x0 + 4u]));
#else
    // sub in [0, 16): half h = sub / 8, positions l0 .. l0 + 3 with l0 = (sub % 8) * 4; each position yields the
    // elements l, l + 32, l + 64 and l + 96 of the half.
    uint h = sub >> 3;
    uint l0 = (sub & 7u) * 4u;
    uint qlA = FN_READ32(bb + h * 64u + l0);
    uint qlB = FN_READ32(bb + h * 64u + l0 + 32u);
    uint qh = FN_READ32(bb + 128u + h * 32u + l0);
    uint scBase = bb + 192u + h * 8u + (l0 >> 4);
    float d = FN_HALF(bb + 208u);
    uint scw0 = FN_READ32(scBase);
    uint scw1 = FN_READ32(scBase + 4u);
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
        uint xb = xOff + m * K4 + x0;
        ACC[m] += d0 * dot(q0, x4[xb]) + d2 * dot(q1, x4[xb + 8u]) + d4 * dot(q2, x4[xb + 16u]) + d6 * dot(q3, x4[xb + 24u]);
    }
#endif
}
