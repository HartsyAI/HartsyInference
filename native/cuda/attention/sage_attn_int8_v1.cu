// SageAttention-v1 INT8 flash attention, register-resident implementation ("v1" — replaces the wmma-based
// v0 in sage_attn_int8.cu, whose UNDEFINED wmma fragment layouts forced O/softmax through shared memory
// with serial per-thread loops: measured 0.17× vs the materialized-cuBLAS baseline, INFERENCE_ACCEL_GRIND
// §H4). Raw mma.sync PTX asm has ARCHITECTURALLY DEFINED lane↔element layouts, which is what enables:
//   • O accumulator resident in registers (16× m16n8 f32 tiles at D=128 — 64 regs/lane),
//   • online-softmax m/l state in registers (each lane owns rows {g, g+8}, g = lane>>2),
//   • the no-shuffle S→P handoff: the m16n8k32-s8 C-fragment layout is register-identical to the
//     m16n8k16-f16 A-fragment layout, so P is a pure f32→f16x2 repack of the softmaxed scores.
//
// Tiling: BR=64 query rows/block (4 warps × 16), BC=32 K/V cols per step, block = 128 threads.
// v1.3: V is pre-transposed to F16 ONCE per forward (sage_v_f16t → [B,H,D,Skv]; per-block re-transposes
// were Sq/64 × redundant), and K8/Vt/ks staging is cp.async.cg 16-byte double-buffered (2 stages fit
// 24.8 KB < the 48 KB default → still 3 blocks/SM). Swizzle granularity is 16 B (cp.async.cg's only size).
// Per K-step per warp: QK^T = 8 n-tiles × 4 k-chunks of mma.m16n8k32.s8 (D=128); PV = 4 k16-chunks ×
// 16 d-tiles of mma.m16n8k16.f16 with f32 accumulate. K8 staged in SMEM (bytes, zero-padded tail);
// V staged TRANSPOSED as F16 (Vt[d][c]) so PV B-fragments are contiguous u32 ld.shared. Padded key
// columns are masked to −inf before the row max (they must not leak into l). Full query-row guarding —
// any Sq dispatches (v0's Sq%32 gate excluded Wan's 3510-token stream); rows ≥ Sq compute on zeros and
// are never stored.
//
// Same I/O contract as v0's sage_attn_int8_f32 (the SageAttnKernelTests parity oracle applies verbatim):
//   out [B,Hq,Sq,D] f32; Q8/K8 [.,.,S,D] s8; qscale [B,Hq,Sq] f32 (attn scale folded); kscale [B,Hq,Skv].
// Grid = (ceil(Sq/64), Hq, B), block = 128. Dynamic SMEM: K8 s8 [BC*D] + Vt f16 [D*BC] + kscale f32 [BC]
// (D=128: 8KB + 16KB + 256B ≈ 24.3KB → 2 blocks/SM within the 48KB default).
//
// Build: nvcc -ptx -arch=sm_80 sage_attn_int8_v1.cu -o sage_attn_int8_v1.ptx  (PTX must say .version 9.0
// — pinned toolchain, INFERENCE_ACCEL_GRIND §H1.1). Reg pressure: check `ptxas -arch=sm_86 -v` ≤ 168.
#include <cuda_fp16.h>

#define BR 64
#define BC 32
#define WARPS 4
#define NEG_INF (-3.402823466e+38f)

// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 — D-chunk of the QK^T tile.
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 — PV tile with F16 ACCUMULATE (2× the f32-acc rate
// on GeForce Ampere, where f16f16f32 mma is half-rate). C/D are 2 regs of packed half2. OVERFLOW HAZARD:
// unnormalized flash-O reaches l·max|v| — the E1 accuracy gate (overflow-adversarial test) decides whether
// this variant ships; it is compiled as separate _f16acc entry points, never silently substituted.
__device__ __forceinline__ void mma_f16acc(unsigned& c0, unsigned& c1,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 — BC-chunk of the PV tile.
__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}



__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src)
{
    const unsigned dst = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }


// skvPad: rows padded to 8 halves (16B cp.async alignment) — and bumped +8 when the result is an EXACT
// power of two ≥ 2048: a pow2 pad makes the Vt row stride (2·skvPad bytes) a pure power of 2, aliasing
// every d-row into the same L2 set group (measured: Skv=8192 ran ~2× slower than its work vs 12288).
__device__ __forceinline__ unsigned sage_skv_pad(unsigned skv)
{
    unsigned p = (skv + 7u) & ~7u;
    if (p >= 2048u && (p & (p - 1u)) == 0u) p += 8u;
    return p;
}

// ── SMEM XOR swizzles (DEEP_KERNEL_OPTIMIZATION §6: mandatory for mma-consumed tiles). Both tiles are
// permuted at 4-element (u32 / 8-byte) granularity WITHIN a row, XORed with the row index, so the
// fragment-read patterns (u32s whose lane addresses previously differed only in the row → same bank)
// spread across all 32 banks. Intra-chunk order is preserved, which the paired reads rely on. ──

// K8 [BC][D] s8: chunk = 16 bytes (cp.async.cg granularity). u32 reads at (row, d) stay inside a chunk
// (d%16 ∈ {0,4,8,12} for kb0; kb1 at d+16 lands in the next chunk).
template<unsigned D>
__device__ __forceinline__ unsigned k8_off(unsigned row, unsigned d)
{
    constexpr unsigned CH = D / 16 - 1;
    const unsigned c = ((d >> 4) ^ (row & CH)) & CH;
    return row * D + c * 16 + (d & 15);
}

// Vt [D][BC] f16: chunk = 8 halves (16 B). u32 reads at (d, key) with key%8 in {0,2,4,6} — pairs stay
// inside one chunk; the +8 partner is the adjacent chunk.
__device__ __forceinline__ unsigned vt_off(unsigned d, unsigned c)
{
    constexpr unsigned CH = BC / 8 - 1;
    const unsigned c8 = ((c >> 3) ^ (d & CH)) & CH;
    return d * BC + c8 * 8 + (c & 7);
}

template<unsigned D, bool F16ACC, bool OUT16, typename OutT>
__device__ __forceinline__ void sage_attn_v1_core(
    OutT* __restrict__ out,               // [B, Hq, Sq, D] (float, or __half when OUT16)
    const signed char* __restrict__ Q8,   // [B, Hq, Sq, D]
    const float* __restrict__ qscale,     // [B, Hq, Sq]
    const signed char* __restrict__ K8,   // [B, Hq, Skv, D]
    const float* __restrict__ kscale,     // [B, Hq, Skv]
    const __half* __restrict__ Vt16,      // [B, Hq, D, Skv]  (pre-transposed F16 — sage_v_f16t)
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv)
{
    const unsigned qtile = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned b = blockIdx.z;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned g = lane >> 2;          // fragment group row: this lane owns rows {g, g+8} of its warp tile
    const unsigned t4 = lane & 3;          // fragment group col index
    const unsigned q0 = qtile * BR;
    if (q0 >= Sq) return;
    const unsigned warpRow0 = q0 + warp * 16;    // this warp's first query row
    constexpr unsigned dTiles = D >> 3;          // m16n8 output tiles across D (16 at D=128, 8 at D=64)
    constexpr unsigned kChunks = D >> 5;         // s8 k32 chunks (4 at D=128, 2 at D=64)

    const size_t headQ = (size_t)Sq * D;
    const size_t headKV = (size_t)Skv * D;
    const signed char* Q8h = Q8 + ((size_t)b * Hq + h) * headQ;
    const signed char* K8h = K8 + ((size_t)b * Hq + h) * headKV;
    const unsigned skvPad = sage_skv_pad(Skv);                   // Vt16 row stride (16B-aligned, anti-aliased)
    const __half* Vth = Vt16 + ((size_t)b * Hq + h) * ((size_t)D * skvPad);   // [D][skvPad] for this head
    const float* qsh = qscale + ((size_t)b * Hq + h) * Sq;
    const float* ksh = kscale + ((size_t)b * Hq + h) * Skv;
    OutT* Oh = out + ((size_t)b * Hq + h) * headQ;

    extern __shared__ unsigned char smemRaw[];
    // Two cp.async stages: [stage][...]. Per stage: K8 [BC][D] s8, Vt [D][BC] f16, ks [BC] f32.
    constexpr unsigned StageBytes = (unsigned)(BC * D + D * BC * sizeof(__half) + BC * sizeof(float));
    auto K8s = [&](unsigned s) { return (signed char*)(smemRaw + (size_t)s * StageBytes); };
    auto Vts = [&](unsigned s) { return (__half*)(smemRaw + (size_t)s * StageBytes + (size_t)BC * D); };
    auto kss = [&](unsigned s) { return (float*)(smemRaw + (size_t)s * StageBytes + (size_t)BC * D + (size_t)D * BC * sizeof(__half)); };

    // ── Q8 fragments: resident for the whole kernel. Rows guarded (≥Sq ⇒ zeros). ──
    // A-frag m16n8k32 layout: reg0 (row g,    d = t4*4 + kchunk*32 + 0..3)   → one u32 from Q8[row][d]
    //                         reg1 (row g+8,  same d)
    //                         reg2 (row g,    d+16), reg3 (row g+8, d+16)
    unsigned qa[kChunks][4];
    {
        const unsigned r0 = warpRow0 + g, r1 = warpRow0 + g + 8;
        #pragma unroll
        for (unsigned kc = 0; kc < kChunks; kc++)
        {
            const unsigned d0 = kc * 32 + t4 * 4;
            qa[kc][0] = r0 < Sq ? *(const unsigned*)(Q8h + (size_t)r0 * D + d0) : 0u;
            qa[kc][1] = r1 < Sq ? *(const unsigned*)(Q8h + (size_t)r1 * D + d0) : 0u;
            qa[kc][2] = r0 < Sq ? *(const unsigned*)(Q8h + (size_t)r0 * D + d0 + 16) : 0u;
            qa[kc][3] = r1 < Sq ? *(const unsigned*)(Q8h + (size_t)r1 * D + d0 + 16) : 0u;
        }
    }
    const float qs0 = (warpRow0 + g) < Sq ? qsh[warpRow0 + g] : 0.0f;
    const float qs1 = (warpRow0 + g + 8) < Sq ? qsh[warpRow0 + g + 8] : 0.0f;

    // ── Online-softmax state + O accumulator, all registers. O tiles indexed [dTile]{c0..c3}:
    //    c0,c1 = (row g,   d = dTile*8 + t4*2 + {0,1}); c2,c3 = (row g+8, same d).
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
    // F32-acc: o[dt][4] floats. F16-acc (E1): f16 mma RATE with f32 accumulation FIDELITY — the mma
    // accumulates into o2 (packed half2) ZEROED EVERY K-STEP (per-step sums span ≤BC keys → no swamping;
    // the naive whole-row f16 accumulator lost mass once Σ ≫ increment ulp — caught by the uniform-
    // attention gate), then drains into the f32 running O right after the step's PV mmas.
    float o[dTiles][4];
    unsigned o2[F16ACC ? dTiles : 1][2];
    #pragma unroll
    for (unsigned dt = 0; dt < dTiles; dt++) { o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f; }

    const unsigned nKsteps = (Skv + BC - 1) / BC;

    // 16-byte cp.async stage loader (K8 + Vt + ks into stage s). Tail chunks fall back to scalar SMEM
    // stores (rare; only the final partial tile). Alignment: all cp.async dst/src are 16B multiples —
    // staged d/c indices are chunk-aligned and Vt16 rows are padded to 8 halves.
    auto stageLoad = [&](unsigned ksIdx, unsigned s)
    {
        const unsigned col0 = ksIdx * BC;
        const unsigned cur = min((unsigned)BC, Skv - col0);
        signed char* k8d = K8s(s);
        __half* vtd = Vts(s);
        float* ksd = kss(s);
        for (unsigned i = threadIdx.x * 16; i < (unsigned)BC * D; i += blockDim.x * 16)
        {
            const unsigned c = i / D, d = i % D;
            if (c < cur) cp_async_16(k8d + k8_off<D>(c, d), K8h + (size_t)(col0 + c) * D + d);
            else *(uint4*)(k8d + k8_off<D>(c, d)) = make_uint4(0u, 0u, 0u, 0u);
        }
        for (unsigned i = threadIdx.x * 8; i < (unsigned)D * BC; i += blockDim.x * 8)
        {
            const unsigned d = i / BC, c = i % BC;
            if (c + 8 <= cur)
            {
                cp_async_16(vtd + vt_off(d, c), Vth + (size_t)d * skvPad + col0 + c);
            }
            else
            {
                for (unsigned j = 0; j < 8; j++)
                    vtd[vt_off(d, c + j)] = (c + j) < cur
                        ? Vth[(size_t)d * skvPad + col0 + c + j] : __float2half(0.0f);
            }
        }
        // ks: plain scalar loads (the per-head kscale base is head*Skv floats — NOT 16B-aligned for odd
        // Skv, which faults cp.async; 32 floats/stage is noise next to the async K8/Vt traffic).
        for (unsigned i = threadIdx.x; i < BC; i += blockDim.x)
            ksd[i] = (i < cur) ? ksh[col0 + i] : 0.0f;
        cp_async_commit();
    };

    unsigned stage = 0;
    stageLoad(0, 0);
    for (unsigned ks = 0; ks < nKsteps; ks++)
    {
        const unsigned c0col = ks * BC;
        const unsigned curBC = min((unsigned)BC, Skv - c0col);

        __syncthreads();   // all threads done READING the buffer the next prefetch will overwrite
        if (ks + 1 < nKsteps) stageLoad(ks + 1, stage ^ 1);
        if (ks + 1 < nKsteps) cp_async_wait<1>(); else cp_async_wait<0>();
        __syncthreads();   // this stage's cp.async writes visible block-wide

        const signed char* K8sh = K8s(stage);
        const __half* Vtsh = Vts(stage);
        const float* kssh = kss(stage);
        stage ^= 1;

        // ── S = Q8 @ K8^T on IMMA, 8 n-tiles of 8 cols. C-frag rows {g,g+8}, cols nt*8 + t4*2 + {0,1}. ──
        int sAcc[8][4];
        #pragma unroll
        for (unsigned nt = 0; nt < (BC >> 3); nt++)
        {
            sAcc[nt][0] = sAcc[nt][1] = sAcc[nt][2] = sAcc[nt][3] = 0;
            #pragma unroll
            for (unsigned kc = 0; kc < kChunks; kc++)
            {
                // B-frag m16n8k32 col-major: b0 = K̄[key = nt*8 + g][d = kc*32 + t4*4 + 0..3] (one u32),
                //                            b1 = same key, d+16.
                const unsigned keyRow = nt * 8 + g;
                const unsigned d0 = kc * 32 + t4 * 4;
                const unsigned kb0 = *(const unsigned*)(K8sh + k8_off<D>(keyRow, d0));
                const unsigned kb1 = *(const unsigned*)(K8sh + k8_off<D>(keyRow, d0 + 16));
                mma_s8(sAcc[nt][0], sAcc[nt][1], sAcc[nt][2], sAcc[nt][3],
                    qa[kc][0], qa[kc][1], qa[kc][2], qa[kc][3], kb0, kb1);
            }
        }

        // ── Dequant + mask + online softmax, all in registers. ──
        float p[8][4];   // dequantized scores → probabilities (f32, C-frag positions)
        float rowMax0 = NEG_INF, rowMax1 = NEG_INF;
        #pragma unroll
        for (unsigned nt = 0; nt < (BC >> 3); nt++)
        {
            const unsigned c01 = nt * 8 + t4 * 2;
            const float ksc0 = c01 < curBC ? kssh[c01] : 0.0f;
            const float ksc1 = (c01 + 1) < curBC ? kssh[c01 + 1] : 0.0f;
            p[nt][0] = c01 < curBC ? (float)sAcc[nt][0] * qs0 * ksc0 : NEG_INF;
            p[nt][1] = (c01 + 1) < curBC ? (float)sAcc[nt][1] * qs0 * ksc1 : NEG_INF;
            p[nt][2] = c01 < curBC ? (float)sAcc[nt][2] * qs1 * ksc0 : NEG_INF;
            p[nt][3] = (c01 + 1) < curBC ? (float)sAcc[nt][3] * qs1 * ksc1 : NEG_INF;
            rowMax0 = fmaxf(rowMax0, fmaxf(p[nt][0], p[nt][1]));
            rowMax1 = fmaxf(rowMax1, fmaxf(p[nt][2], p[nt][3]));
        }
        // quad-reduce the row maxima (lanes g*4..g*4+3 hold the same rows)
        for (unsigned off = 1; off <= 2; off <<= 1)
        {
            rowMax0 = fmaxf(rowMax0, __shfl_xor_sync(0xffffffffu, rowMax0, off));
            rowMax1 = fmaxf(rowMax1, __shfl_xor_sync(0xffffffffu, rowMax1, off));
        }

        const float mNew0 = fmaxf(m0, rowMax0), mNew1 = fmaxf(m1, rowMax1);
        // log2-domain softmax: the quant prologue folds log2(e) into qscale, so scores arrive pre-scaled
        // and exp2f replaces expf (drops one multiply per score; MUFU.EX2 is the native SFU op).
        // all-masked tile (or empty row): keep NEG_INF states from corrupting corr via exp2(-inf - -inf)
        const float corr0 = (mNew0 == NEG_INF) ? 1.0f : exp2f(m0 - mNew0);
        const float corr1 = (mNew1 == NEG_INF) ? 1.0f : exp2f(m1 - mNew1);

        float rowSum0 = 0.0f, rowSum1 = 0.0f;
        #pragma unroll
        for (unsigned nt = 0; nt < (BC >> 3); nt++)
        {
            p[nt][0] = (p[nt][0] == NEG_INF) ? 0.0f : exp2f(p[nt][0] - mNew0);
            p[nt][1] = (p[nt][1] == NEG_INF) ? 0.0f : exp2f(p[nt][1] - mNew0);
            p[nt][2] = (p[nt][2] == NEG_INF) ? 0.0f : exp2f(p[nt][2] - mNew1);
            p[nt][3] = (p[nt][3] == NEG_INF) ? 0.0f : exp2f(p[nt][3] - mNew1);
            rowSum0 += p[nt][0] + p[nt][1];
            rowSum1 += p[nt][2] + p[nt][3];
        }
        for (unsigned off = 1; off <= 2; off <<= 1)
        {
            rowSum0 += __shfl_xor_sync(0xffffffffu, rowSum0, off);
            rowSum1 += __shfl_xor_sync(0xffffffffu, rowSum1, off);
        }
        l0 = l0 * corr0 + rowSum0;
        l1 = l1 * corr1 + rowSum1;
        m0 = mNew0;
        m1 = mNew1;

        // rescale the f32 running O by corr (both variants — o2 is per-step transient in F16ACC mode)
        #pragma unroll
        for (unsigned dt = 0; dt < dTiles; dt++)
        {
            o[dt][0] *= corr0; o[dt][1] *= corr0;
            o[dt][2] *= corr1; o[dt][3] *= corr1;
        }
        if constexpr (F16ACC)
        {
            #pragma unroll
            for (unsigned dt = 0; dt < dTiles; dt++) { o2[dt][0] = 0u; o2[dt][1] = 0u; }
        }

        // ── P @ V on f16 tensor cores. The S→P handoff is pure register repacking (layout identity):
        //    PV k16-chunk kc covers P n-tiles {2kc, 2kc+1}; A-frag = {p[2kc] rows g/g+8, p[2kc+1] rows g/g+8}
        //    packed as f16x2. B-frag from Vt: b0 = Vt[d = dTile*8 + g][key = kc*16 + t4*2 .. +1] (one u32),
        //    b1 = same d, key+8. ──
        // f16-acc headroom: pack P pre-scaled by 1/16 (exponent shift, exact for p in [0,1]); the
        // epilogue multiplies the reciprocal-l by 16 to compensate. f32-acc packs P unscaled.
        const float pScale = F16ACC ? (1.0f / 16.0f) : 1.0f;
        #pragma unroll
        for (unsigned kc = 0; kc < (BC >> 4); kc++)
        {
            const __half2 h00 = __floats2half2_rn(p[2 * kc][0] * pScale, p[2 * kc][1] * pScale);   // row g, k-cols t4*2,t4*2+1
            const __half2 h01 = __floats2half2_rn(p[2 * kc][2] * pScale, p[2 * kc][3] * pScale);       // row g+8
            const __half2 h10 = __floats2half2_rn(p[2 * kc + 1][0] * pScale, p[2 * kc + 1][1] * pScale); // row g, k-cols +8
            const __half2 h11 = __floats2half2_rn(p[2 * kc + 1][2] * pScale, p[2 * kc + 1][3] * pScale); // row g+8
            const unsigned a0 = *(const unsigned*)&h00;
            const unsigned a1 = *(const unsigned*)&h01;
            const unsigned a2 = *(const unsigned*)&h10;
            const unsigned a3 = *(const unsigned*)&h11;
            // PV B-fragments: manual u32 ld.shared (an ldmatrix.x4 variant measured 1.5% SLOWER —
            // loads are already latency-hidden behind the mma chains post-cp.async; the per-lane
            // swizzled-address ALU for ldmatrix outweighed its instruction savings. 2026-07-22.)
            #pragma unroll
            for (unsigned dt = 0; dt < dTiles; dt++)
            {
                const unsigned dRow = dt * 8 + g;                    // Vt row = output d index
                const unsigned key0 = kc * 16 + t4 * 2;
                const unsigned vb0 = *(const unsigned*)(Vtsh + vt_off(dRow, key0));
                const unsigned vb1 = *(const unsigned*)(Vtsh + vt_off(dRow, key0 + 8));
                if constexpr (F16ACC) mma_f16acc(o2[dt][0], o2[dt][1], a0, a1, a2, a3, vb0, vb1);
                else mma_f16(o[dt][0], o[dt][1], o[dt][2], o[dt][3], a0, a1, a2, a3, vb0, vb1);
            }
        }
        // drain this K-step's f16 tile sums into the f32 running O (F16ACC: preserves fidelity at any l)
        if constexpr (F16ACC)
        {
            #pragma unroll
            for (unsigned dt = 0; dt < dTiles; dt++)
            {
                const float2 f0 = __half22float2(*(const __half2*)&o2[dt][0]);
                const float2 f1 = __half22float2(*(const __half2*)&o2[dt][1]);
                o[dt][0] += f0.x; o[dt][1] += f0.y;
                o[dt][2] += f1.x; o[dt][3] += f1.y;
            }
        }
    }

    // ── Epilogue: O /= l, guarded stores in C-frag positions. ──
    const float invScale = F16ACC ? 16.0f : 1.0f;   // undo the P pre-scale (see PV pack)
    const float inv0 = l0 > 0.0f ? invScale / l0 : 0.0f;
    const float inv1 = l1 > 0.0f ? invScale / l1 : 0.0f;
    const unsigned r0 = warpRow0 + g, r1 = warpRow0 + g + 8;
    #pragma unroll
    for (unsigned dt = 0; dt < dTiles; dt++)
    {
        const unsigned d0 = dt * 8 + t4 * 2;
        if (r0 < Sq)
        {
            Oh[(size_t)r0 * D + d0] = (OutT)(o[dt][0] * inv0);
            Oh[(size_t)r0 * D + d0 + 1] = (OutT)(o[dt][1] * inv0);
        }
        if (r1 < Sq)
        {
            Oh[(size_t)r1 * D + d0] = (OutT)(o[dt][2] * inv1);
            Oh[(size_t)r1 * D + d0 + 1] = (OutT)(o[dt][3] * inv1);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 3) sage_attn_int8_v1_d128_f32(
    float* __restrict__ out, const signed char* __restrict__ Q8, const float* __restrict__ qscale,
    const signed char* __restrict__ K8, const float* __restrict__ kscale, const __half* __restrict__ Vt16,
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv)
{
    sage_attn_v1_core<128, false, false, float>(out, Q8, qscale, K8, kscale, Vt16, B, Hq, Sq, Skv);
}

extern "C" __global__ void __launch_bounds__(128, 3) sage_attn_int8_v1_d64_f32(
    float* __restrict__ out, const signed char* __restrict__ Q8, const float* __restrict__ qscale,
    const signed char* __restrict__ K8, const float* __restrict__ kscale, const __half* __restrict__ Vt16,
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv)
{
    sage_attn_v1_core<64, false, false, float>(out, Q8, qscale, K8, kscale, Vt16, B, Hq, Sq, Skv);
}

// ── V pre-transpose+cast, ONCE per forward: [B,Hq,Skv,D] f32 → [B,Hq,D,skvPad] f16 (skvPad = Skv
// rounded up to 8 so every 8-half chunk is a 16B-aligned cp.async source; pad halves are zero).
// Grid-stride over the OUTPUT (writes coalesced along skvPad; reads strided by D — one pass, per-forward
// cost is amortized over Sq/64 × query-tile blocks that previously each re-transposed V).
extern "C" __global__ void sage_v_f16t(
    __half* __restrict__ vt,              // [B, Hq, D, skvPad]
    const float* __restrict__ V,          // [B, Hq, Skv, D]
    unsigned int B, unsigned int Hq, unsigned int Skv, unsigned int D)
{
    const unsigned skvPad = sage_skv_pad(Skv);
    const size_t total = (size_t)B * Hq * D * skvPad;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const unsigned s = (unsigned)(i % skvPad);
        const size_t rest = i / skvPad;
        const unsigned d = (unsigned)(rest % D);
        const size_t bh = rest / D;
        vt[i] = (s < Skv) ? __float2half(V[(bh * Skv + s) * D + d]) : __float2half(0.0f);
    }
}

extern "C" __global__ void __launch_bounds__(128, 3) sage_attn_int8_v1_d128_f16acc_f32(
    float* __restrict__ out, const signed char* __restrict__ Q8, const float* __restrict__ qscale,
    const signed char* __restrict__ K8, const float* __restrict__ kscale, const __half* __restrict__ Vt16,
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv)
{
    sage_attn_v1_core<128, true, false, float>(out, Q8, qscale, K8, kscale, Vt16, B, Hq, Sq, Skv);
}

extern "C" __global__ void __launch_bounds__(128, 3) sage_attn_int8_v1_d64_f16acc_f32(
    float* __restrict__ out, const signed char* __restrict__ Q8, const float* __restrict__ qscale,
    const signed char* __restrict__ K8, const float* __restrict__ kscale, const __half* __restrict__ Vt16,
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv)
{
    sage_attn_v1_core<64, true, false, float>(out, Q8, qscale, K8, kscale, Vt16, B, Hq, Sq, Skv);
}

extern "C" __global__ void __launch_bounds__(128, 3) sage_attn_int8_v1_d128_f16io(
    __half* __restrict__ out, const signed char* __restrict__ Q8, const float* __restrict__ qscale,
    const signed char* __restrict__ K8, const float* __restrict__ kscale, const __half* __restrict__ Vt16,
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv)
{
    sage_attn_v1_core<128, true, true, __half>(out, Q8, qscale, K8, kscale, Vt16, B, Hq, Sq, Skv);
}

extern "C" __global__ void __launch_bounds__(128, 3) sage_attn_int8_v1_d64_f16io(
    __half* __restrict__ out, const signed char* __restrict__ Q8, const float* __restrict__ qscale,
    const signed char* __restrict__ K8, const float* __restrict__ kscale, const __half* __restrict__ Vt16,
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv)
{
    sage_attn_v1_core<64, true, true, __half>(out, Q8, qscale, K8, kscale, Vt16, B, Hq, Sq, Skv);
}

// F16-source V pre-transpose (no cast — pure [B,Hq,Skv,D]→[B,Hq,D,skvPad] gather + zero-pad).
extern "C" __global__ void sage_v_f16t_h(
    __half* __restrict__ vt, const __half* __restrict__ V,
    unsigned int B, unsigned int Hq, unsigned int Skv, unsigned int D)
{
    const unsigned skvPad = sage_skv_pad(Skv);
    const size_t total = (size_t)B * Hq * D * skvPad;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        const unsigned s = (unsigned)(i % skvPad);
        const size_t rest = i / skvPad;
        const unsigned d = (unsigned)(rest % D);
        const size_t bh = rest / D;
        vt[i] = (s < Skv) ? V[(bh * Skv + s) * D + d] : __float2half(0.0f);
    }
}
