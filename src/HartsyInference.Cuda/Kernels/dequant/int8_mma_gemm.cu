// INT8 tensor-core GEMM with the W8A8 dequant FUSED INTO THE EPILOGUE.
//
// Why this exists: the cuBLASLt path computes D_i32[M,N] and a second kernel reads it back to apply
// actScale[row]*wScale[col] (+bias, +act). At LTX-2.5's shapes that accumulator round trip is ~203 GB/step,
// and cuBLASLt provably cannot fuse it — asked directly, it accepts ONLY int32 D with a scalar alpha for
// int8 operands (every {F16 D, device-vector alpha} combination returns rc=15). So the epilogue has to live
// inside our own mma kernel. comfy-kitchen does the same thing through a CUTLASS epilogue.
//
// D[M,N] = act(actScale[m] * wScale[n] * sum_k A[m,k]*B[n,k] + bias[n])
//   A [M,K] row-major int8 (per-row dynamically quantized activation)
//   B [N,K] row-major int8 (the stored weight; row-major [N,K] IS mma's ".col" B operand)
//
// Tiling: block 128(M) x 256(N) x 64(K), 8 warps as 2(M) x 4(N), warp tile 64 x 64 = 4x8 mma m16n8k32 tiles
// (128 int32 accumulator registers/thread). STAGES shared-memory stages, filled with cp.async.
//
// Measured on a 4090 against cuBLASLt-GEMM + w8a8_dequant_bias, bit-exact (max abs 0) at every shape including
// ragged-M, COLD L2 (see ColdBuffers in Int8MmaGemmTests — reusing one resident weight flatters this kernel and
// inverted a conclusion once): attn_qkvo 4992x4096x4096 437.5 TOPS = +11.9% vs the pair, ffn_up 4992x16384x4096
// 411.0 = +3.5%, ffn_down 4992x4096x16384 465.1 = -22.0%. Bare cuBLASLt is 565.7/575.3/664.9, so the mainloop
// still runs ~23% behind it on the square shape. CudaBackend.UseFusedMmaGemm admits only the winning regime.
//
// The swizzled layout is worth **-18.4 ms/step** end-to-end against the padded one (ltx25_ab.sh, 4 interleaved
// reps, all four pairs same sign, paired t = 7.53), on top of the fused kernel's own -19.7.
//
// TWO ENTRY POINTS, and they are a MEASUREMENT CONTROL, not an abstraction:
//   int8_mma_gemm_dequant_f16      swizzled, unpadded shared layout (ships)
//   int8_mma_gemm_dequant_f16_pad  the BK+16 padded layout it replaced (HARTSY_INT8_MMA_SWIZZLE=0)
// The bodies are duplicated rather than templated on purpose: a template parameter threaded through one body
// perturbs register allocation on BOTH instantiations, and the padded arm's job is to reproduce the shipped
// kernel's SASS exactly so an A/B against it means something. Keep them separate. Their dynamic shared sizes
// DIFFER (73,728 vs 92,160 B) — CudaKernels launches each with its own, never the larger of the two.
//
// WHY 128x256 AND NOT 128x128 — arithmetic intensity, and it was the single biggest win (288 -> 341 TOPS).
// A 128x128 block reads 16 KB of operands per k-tile to do 2.1 MFLOP — 131 flop/byte — so sustaining the 4090's
// 660 TOPS would demand ~5 TB/s out of L2, at its ceiling and far past HBM. 128x256 reads 24 KB for 4.2 MFLOP,
// and per unit of OUTPUT the A tile is reused twice as much. This is comfy-kitchen's shipped tile for these
// shapes. The cost is that 128 accumulator registers (195 total, no spill) force one block per SM.
//
// THE LIMITER IS **NOT** L2 BANDWIDTH — and the experiment that proved that is the swizzle below, so read this
// before proposing another traffic fix. The control throughout is cuBLASLt at attn_qkvo, which demangles to
// cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn_align16 — the SAME tile, warp shape, instruction, swizzle
// and stage count as ours, so it is like-for-like and there is nothing left to copy configuration-wise.
//
// The 08-13 reading was that duration ratio 1.386x tracked L2 sector ratio 1.424x, therefore L2-bound,
// therefore the kernel goes faster only by moving fewer bytes. Per-SASS attribution put 100% of the excess on
// ONE instruction, LDGSTS.E.BYPASS.128 (the cp.async operand loads), and the swizzle below removed all of it:
// L2 sectors 47.36M -> 32.02M (BELOW cuBLASLt's 33.24M), global sectors 38.46M -> 30.79M (within 0.4% of its
// 30.67M), L2 throughput 85.25% -> 50.42% against its 73.01%. The kernel got **4.6%** faster. So the ratio was
// a coincidence: we are now less L2-loaded than the control and still 1.36x slower than it.
//
// What survives as the asymmetry, for whoever picks this up: stall mio_throttle 2.27 against cuBLASLt's 0.33,
// while stall long_scoreboard is LOWER than its (0.17 vs 0.34) and achieved occupancy is identical (16.66 vs
// 16.75). That is an issue-side stall on the memory-IO pipe, not bandwidth and not latency. It is NOT the
// epilogue's bank conflicts either — zeroing those moved it only 2.27 -> 2.21 (see the epilogue).
//
// AND THE WASTE WAS IN THE SHARED DESTINATION, NOT THE GLOBAL SOURCE. This is worth stating precisely because
// the obvious reading of "global LD sectors/request 18.88 against an ideal 16.00" is that the loads themselves
// straddle cache lines, and they do not: load_tile's global address is gr*K + k0 + (tid&3)*16 with K and k0
// both multiples of 64, so every 64-byte run is 64-byte aligned and occupies exactly 2 sectors of one 128-byte
// line — 16 per warp request, already ideal, and INDEPENDENT of the shared stride. The pad could not have been
// splitting lines. What it split was the INSTRUCTION: LDGSTS is one op whose destination addressing is the
// shared side, L1TEX serializes it into wavefronts by shared-bank conflict, and each wavefront runs its own
// global tag lookup, so the same sectors get re-tagged and counted again. Hence 18.88 sectors/request at only
// 1.18x, against 1.50x on total sectors — the residual is extra requests, i.e. replays, not extra bytes (the
// loads were 97.5% L2 hits throughout). The prediction that follows is falsifiable and is why this layout
// exists: killing the shared-store conflicts must pull sectors/request back toward 16.0 WITHOUT ONE GLOBAL
// ADDRESS CHANGING. If you are editing load_tile's global index math you are past the diagnosis.
//
// THE SHARED LAYOUT — unpadded 64-byte rows, 16-byte vectors XOR-permuted within each row:
//     physical byte offset of vector v (0..3) of tile row r  =  r*64 + ((v ^ ((r >> 1) & 3)) << 4)
// This is CUTLASS's TensorOpMultiplicandCrosswise<8,64> permutation, arrived at independently and then checked
// against it: that layout's kFactor is kTileShapeContiguous*kElementsPerAccess/Crosswise = 8*16/64 = 2, it folds
// two strided rows into each 128-byte line, and permutes with `partition_contiguous_residual ^
// (partition_strided_residual % 4)` — which at this shape IS `v ^ ((r/2) & 3)`.
//
// WHY (r >> 1) & 3 AND NOT r & 3, which is the form you write first: a 64-byte row means the bank pattern
// repeats every TWO rows, so an ldmatrix reading 8 consecutive rows sees only 2 distinct 64-byte phases and
// needs the other 4 slots from the permutation. r & 3 has period 4 and collides r=0 with r=4. (r >> 1) & 3
// gives the 8 rows the 8 distinct (r&1, v') slots of a 128-byte window. All three access patterns are then
// conflict-free, which is the hard constraint — the pad bought ldmatrix 1.00x and the swizzle must not sell it:
//   cp.async store  a warp writes rows 8w..8w+7 x all 4 vectors; the XOR permutes within a row, so the 512
//                   bytes are still covered exactly once -> 4 wavefronts, the minimum.
//   A ldmatrix.x4   each 8x8 matrix is 8 consecutive rows at one vector; slots (r&1)*64 + (v ^ ((r>>1)&3))*16
//                   are 8 distinct 16-byte slots of a 128-byte window. (R0 = warpM*64 + i*16 is 0 mod 16, so
//                   every matrix starts on the same phase.)
//   B ldmatrix.x2   same, 8 rows from R0 = warpN*64 + j*8 (0 mod 8).
//
// The epilogue slab has its own, separate 8-way conflict (ncu: 8.00x excess wavefronts on its STS). It is a
// different buffer with a different geometry and is fixed independently — see the epilogue.
//
// Occupancy is 16.7% (8 warps, one block/SM) and ncu flags an "83% local speedup" against it — IGNORE THAT
// for this kernel. The 128 accumulator registers that force one block per SM are the same thing that buys the
// arithmetic intensity; raising occupancy means a smaller tile, which raises L2 traffic, which is the actual
// constraint. ncu's occupancy advice does not know that.
//
// RULED OUT as limiters, each measured against the cuBLASLt control, so do not re-chase:
//   - Shared-load instruction count. ldmatrix cut 24 fragment loads per k-step to 8 and bought 1-6%.
//   - Occupancy per the paragraph above. cuBLASLt wins at an IDENTICAL achieved occupancy (16.34 vs 16.66).
//   - A 4th pipeline stage. stall long_scoreboard is 0.17 of 15.8 cycles — there is no global latency left to
//     hide — and the cuBLASLt control wins at 3 stages.
//   - BK=128. An earlier version of this comment proposed it on the reasoning that a 64-byte run inherently
//     costs 3 sectors; the control runs BK=64 at exactly 1.00x, which refutes the premise outright.
//   - 128x128 (see the tile note above; measured 288 against 413).
//   - Stream-K. `nm -DC` on comfy-kitchen finds zero StreamK symbols; all 576 Sm80 int8 kernels use
//     GemmIdentityThreadblockSwizzle<1>.
//
//   - Shared-store bank conflicts, AS A COUNTER. All 2,236,416 of them live in the epilogue slab, not in the
//     operand path, and zeroing them is measurably worth nothing — the full accounting and numbers are in the
//     epilogue comment. Note this cuts both ways and neither direction of the easy argument was sound: the
//     08-13 comment dismissed conflicts BECAUSE L1/TEX had headroom (wrong reasoning, right answer for the
//     epilogue, wrong answer for the operand pad), and the 08-14 profile promoted them to the headline defect
//     BECAUSE the counter was large (also wrong). A conflict count is not a stall.
//
// The epilogue, by contrast, was worth ~20% and was nearly skipped on the reasoning that "cuBLASLt writes twice
// these bytes as int32 and is still faster". That compares bytes moved; the right metric is SECTORS TOUCHED —
// its int32 store is perfectly coalesced and mma's fragment layout is not. See the epilogue for the fix.
//
// STAGES > 2 needs more than the 48 KB default of dynamic shared, so the launcher must opt in via
// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES — and must ask for EXACTLY what is used, never the SM ceiling,
// or the driver sizes occupancy against the ceiling and silently drops a block per SM. Keep STAGES and
// CudaKernels.Int8MmaSharedBytes / Int8MmaSharedBytesPad in step.

#include <cuda_fp16.h>

#define BM 128
#define BN 256
#define BK 64
#define STAGES 3

// Swizzled (shipping) layout: unpadded 64-byte rows.
#define SW_STAGE_A_BYTES (BM * BK)
#define SW_STAGE_B_BYTES (BN * BK)
// Physical byte offset of 16-byte vector `v` of tile row `r`. See the layout note above.
#define SW_OFF(r, v) ((r) * BK + ((((v) ^ (((r) >> 1) & 3))) << 4))

// Padded control layout: the BK+16 stride this replaced.
#define PAD_STRIDE (BK + 16)
#define PAD_STAGE_A_BYTES (BM * PAD_STRIDE)
#define PAD_STAGE_B_BYTES (BN * PAD_STRIDE)

__device__ __forceinline__ unsigned smem_u32(const void* p)
{
    return (unsigned)__cvta_generic_to_shared(p);
}

// 16-byte global->shared copy that bypasses L1 (.cg): these tiles are streamed once each.
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src, bool pred)
{
    if (pred)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
    else
        asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};\n" ::"r"(dst));   // OOB rows contribute zeros
}

// Stages one TILE_ROWS x 64 int8 tile of a [rows, K] row-major matrix into shared. Each row is 4 vectors of
// 16 B, so 256 threads cover TILE_ROWS*4/256 vectors each — 2 for the 128-row A tile, 4 for the 256-row B tile.
// This thread map (4 threads along K, 8 rows per warp) is CUTLASS's PitchLinearWarpRakedThreadMap arrangement
// and is IDENTICAL in both layouts; only the shared destination differs.
template <int TILE_ROWS>
__device__ __forceinline__ void load_tile_sw(unsigned sBase, const signed char* g, int row0, int rows,
                                             int k0, int K, int tid)
{
    #pragma unroll
    for (int i = 0; i < TILE_ROWS / 64; i++)
    {
        int idx = tid + i * 256;
        int r = idx >> 2;                 // 0..TILE_ROWS-1
        int v = idx & 3;                  // 16-byte vector along K
        int gr = row0 + r;
        cp_async16(sBase + SW_OFF(r, v), g + (size_t)gr * K + k0 + v * 16, gr < rows);
    }
}

template <int TILE_ROWS>
__device__ __forceinline__ void load_tile_pad(unsigned sBase, const signed char* g, int row0, int rows,
                                              int k0, int K, int tid)
{
    #pragma unroll
    for (int i = 0; i < TILE_ROWS / 64; i++)
    {
        int idx = tid + i * 256;
        int r = idx >> 2;
        int c = (idx & 3) * 16;
        int gr = row0 + r;
        cp_async16(sBase + r * PAD_STRIDE + c, g + (size_t)gr * K + k0 + c, gr < rows);
    }
}

// ── ldmatrix fragment loads ────────────────────────────────────────────────
// One `ldmatrix.x4` replaces the 4 scalar ld.shared.b32 an A fragment needed, and one `.x2` the 2 a B fragment
// needed: 24 shared loads per k-step become 8.
//
// ldmatrix moves 8x8 tiles of 16-bit elements, i.e. 8 rows x 16 BYTES, which is exactly how m16n8k32 wants its
// int8 operands laid out, so no repack is needed — only the right per-lane address. Lane L addresses row L%8 of
// matrix L/8; mapping that onto mma's operand layout gives:
//   A (16 rows x 32 int8): matrices are (rows 0-7, vec v), (rows 8-15, vec v), (rows 0-7, vec v+1),
//   (rows 8-15, vec v+1) -> row = L%16, vec = kk/16 + L/16, and {r0..r3} land directly in mma's {a0..a3}.
//   B (8 rows x 32 int8): two matrices -> row = L%8, vec = kk/16 + (L/8)&1.
// The swizzle is applied to (row, vec) at this point, so the DATA ldmatrix returns is unchanged — only where
// it was parked. That is why the mma operand registers need no rework.
__device__ __forceinline__ void ldmatrix_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned addr, unsigned& r0, unsigned& r1)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ float gelu_tanh(float x)
{
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x3)));
}

// ══ SHIPPING KERNEL: unpadded XOR-swizzled operand layout ═══════════════════════════════════════════════
// actMode: 0 = none, 1 = gelu-tanh (matches w8a8_dequant_bias's actMode).
extern "C" __global__ __launch_bounds__(256, 1) void int8_mma_gemm_dequant_f16(
    __half* __restrict__ D,
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ actScale,
    const float* __restrict__ wScale,
    const float* __restrict__ bias,
    unsigned int M, unsigned int N, unsigned int K, unsigned int actMode)
{
    extern __shared__ char smem[];
    unsigned sA0 = smem_u32(smem);
    unsigned sB0 = sA0 + STAGES * SW_STAGE_A_BYTES;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;      // mma fragment addressing
    const int warpM = warp >> 2, warpN = warp & 3;  // 2 x 4 warps -> 2*64 = BM, 4*64 = BN

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0;

    const int kTiles = K / BK;

    // Prologue: fill STAGES-1 stages so the mainloop always has a tile in flight behind the one it computes.
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++)
    {
        if (s < kTiles)
        {
            load_tile_sw<BM>(sA0 + s * SW_STAGE_A_BYTES, A, m0, M, s * BK, K, tid);
            load_tile_sw<BN>(sB0 + s * SW_STAGE_B_BYTES, B, n0, N, s * BK, K, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int kt = 0; kt < kTiles; kt++)
    {
        int cur = kt % STAGES;
        // Leaves STAGES-2 groups outstanding, i.e. waits for exactly the group holding tile kt.
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        // Also the barrier that makes it safe to overwrite the stage read STAGES-1 iterations ago, which is
        // the one the prefetch below targets.
        __syncthreads();

        int fetch = kt + STAGES - 1;
        if (fetch < kTiles)
        {
            int fs = fetch % STAGES;
            load_tile_sw<BM>(sA0 + fs * SW_STAGE_A_BYTES, A, m0, M, fetch * BK, K, tid);
            load_tile_sw<BN>(sB0 + fs * SW_STAGE_B_BYTES, B, n0, N, fetch * BK, K, tid);
        }
        // Committed unconditionally: the wait above counts groups, so a skipped tail fetch still needs one.
        asm volatile("cp.async.commit_group;\n" ::);

        unsigned aStage = sA0 + cur * SW_STAGE_A_BYTES;
        unsigned bStage = sB0 + cur * SW_STAGE_B_BYTES;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 32)
        {
            const int v0 = kk >> 4;        // first of the two 16-byte vectors this k-step reads
            unsigned bfrag[8][2];
            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                int r = warpN * 64 + j * 8 + (lane & 7);
                ldmatrix_x2(bStage + SW_OFF(r, v0 + ((lane >> 3) & 1)), bfrag[j][0], bfrag[j][1]);
            }
            unsigned afrag[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int r = warpM * 64 + i * 16 + (lane & 15);
                ldmatrix_x4(aStage + SW_OFF(r, v0 + (lane >> 4)),
                            afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                unsigned a0 = afrag[i][0], a1 = afrag[i][1], a2 = afrag[i][2], a3 = afrag[i][3];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
                }
            }
        }
        // No trailing barrier: the next iteration's top-of-loop __syncthreads() is what protects this stage
        // from the prefetch that overwrites it, and a second barrier here would only serialize the mainloop.
    }

    // ── Fused dequant epilogue: the int32 accumulator never leaves registers ──
    // mma leaves each thread holding COLUMN PAIRS (tig*2, tig*2+1) of a row, so a warp covered 8 rows x 16
    // contiguous bytes per global store — half of a 32-byte sector. Staging a 16-row x 256-col slab through
    // shared (8 KB, reusing the mainloop's now-dead stage buffer) and re-emitting it lets a warp write 512
    // contiguous bytes per instruction. That was worth 341 -> 413 TOPS. The 16 rows of a slab are two groups
    // of 8 (warpM splits them 64 apart), so the slab is indexed [warpM*8 + gid] and the writer re-derives the
    // real row from that.
    //
    // THIS SLAB STORE OWNS EVERY SHARED-STORE BANK CONFLICT IN THE KERNEL, AND FIXING THEM IS WORTH NOTHING.
    // Both halves of that are measured, and the second is why the code below is the plain version.
    //
    // The attribution first, because it corrects the profile's reading. A slab row is BN halves = 512 B, so
    // every row starts on bank 0 and the 8 lanes sharing a `j` (gid 0..7, one column) land in the same 4 banks
    // — an 8-way conflict. The arithmetic closes to the digit: 624 blocks x 8 warps x (4 i x 2 half x 8 j) =
    // 319,488 ideal wavefronts, x8 = 2,555,904 measured, conflicts = 7/8 of that = 2,236,416, which IS the whole
    // kernel's reported conflict count. The cp.async operand stores contribute exactly none of it — LDGSTS does
    // not retire through the LSU shared-store path these counters watch. So the profile's "2.24M shared ST
    // conflicts" and its "LDGSTS 1.50x sectors" were two unrelated defects that looked like one.
    //
    // Then the negative. Permuting the 16-byte vector index by the slab row (physical = logical ^ (slabRow & 7))
    // is conflict-free by construction and measured exactly that: conflicts 2,236,416 -> 0 and wavefronts
    // 2,555,904 -> 319,488, the ideal, and better than cuBLASLt's 638,976. It bought NO TIME — attn_qkvo 437.5
    // -> 435.9 TOPS, ffn_up 411.0 -> 403.2, kernel duration 417.2 -> 423.0 us, and stall mio_throttle moved only
    // 2.27 -> 2.21 — so it was reverted rather than shipped for a clean-looking counter. The lesson is the one
    // this file keeps re-learning from the other direction: a conflict count is not a stall, and L1/TEX has
    // enough headroom here (SM throughput 60%, L2 50% after the operand fix) to absorb an 8-way store conflict
    // in an epilogue that is a few percent of the kernel. Do not re-try this without a NEW reason.
    __half* slab = (__half*)smem;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int half = 0; half < 2; half++)
        {
            int row = m0 + warpM * 64 + i * 16 + gid + half * 8;
            float as = row < (int)M ? actScale[row] : 0.0f;
            int slabRow = warpM * 8 + gid;
            __syncthreads();   // previous slab fully drained before overwriting it
            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                int cs = warpN * 64 + j * 8 + tig * 2;
                int col = n0 + cs;
                // Spelled as an explicit fma over (acc*rowScale) so the contraction matches what nvrtc emits for
                // w8a8_dequant_bias's `v = (float)d * rowScale * wScale; v += bias`. Left to the compiler the two
                // kernels contract differently and the F32 results diverge by 1 ulp, which lands ~2 outputs per
                // 500k on the far side of an F16 rounding boundary — enough to break an exact-equality gate.
                float t0 = (float)acc[i][j][half * 2 + 0] * as;
                float t1 = (float)acc[i][j][half * 2 + 1] * as;
                float v0 = bias ? fmaf(t0, wScale[col], bias[col]) : t0 * wScale[col];
                float v1 = bias ? fmaf(t1, wScale[col + 1], bias[col + 1]) : t1 * wScale[col + 1];
                if (actMode == 1u) { v0 = gelu_tanh(v0); v1 = gelu_tanh(v1); }
                *(__half2*)(slab + slabRow * BN + cs) = __halves2half2(__float2half(v0), __float2half(v1));
            }
            __syncthreads();
            // 16 rows x 256 halves, emitted as TWO passes of one 16-byte int4 per lane — NOT one pass of two
            // int4s per lane. Both move the same bytes; only this one is coalesced: with 16 lanes per row each
            // writing 32 B as two 16-B stores, every store instruction strides 32 B and touches each 32-byte
            // sector half-used. Here a warp's 32 lanes cover one whole 512-byte output row in one instruction.
            #pragma unroll
            for (int p = 0; p < 2; p++)
            {
                int slot = tid + p * 256;          // 0..511
                int wr = slot >> 5;                // 0..15  slab row, 32 lanes per row
                int wc = (slot & 31) * 8;          // 0..248 halves — 16 B per lane, contiguous across the warp
                int outRow = m0 + (wr >> 3) * 64 + i * 16 + half * 8 + (wr & 7);
                if (outRow < (int)M)
                    *(int4*)(D + (size_t)outRow * N + n0 + wc) = *(const int4*)(slab + wr * BN + wc);
            }
        }
    }
}

// ══ CONTROL KERNEL: the BK+16 padded layout, kept verbatim as the A/B baseline ══════════════════════════
// Reached only via HARTSY_INT8_MMA_SWIZZLE=0. Do not "clean this up" against the kernel above — its value is
// that it is byte-for-byte the code the -19.7 ms/step measurement was taken on.
//
// The pad exists because at the natural 64-byte stride the 8 rows a warp reads for one A fragment land on only
// 8 distinct banks (row*16+tig mod 32 repeats every 2 rows), a 4-way conflict on every fragment load. 80 bytes
// makes the 8 rows hit 8 distinct bank groups and keeps the 16-byte cp.async alignment. It fixes ldmatrix and
// breaks the cp.async STORE — which is the trade the swizzled layout above does not have to make.
extern "C" __global__ __launch_bounds__(256, 1) void int8_mma_gemm_dequant_f16_pad(
    __half* __restrict__ D,
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ actScale,
    const float* __restrict__ wScale,
    const float* __restrict__ bias,
    unsigned int M, unsigned int N, unsigned int K, unsigned int actMode)
{
    extern __shared__ char smem[];
    unsigned sA0 = smem_u32(smem);
    unsigned sB0 = sA0 + STAGES * PAD_STAGE_A_BYTES;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int warpM = warp >> 2, warpN = warp & 3;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            #pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0;

    const int kTiles = K / BK;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++)
    {
        if (s < kTiles)
        {
            load_tile_pad<BM>(sA0 + s * PAD_STAGE_A_BYTES, A, m0, M, s * BK, K, tid);
            load_tile_pad<BN>(sB0 + s * PAD_STAGE_B_BYTES, B, n0, N, s * BK, K, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int kt = 0; kt < kTiles; kt++)
    {
        int cur = kt % STAGES;
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        __syncthreads();

        int fetch = kt + STAGES - 1;
        if (fetch < kTiles)
        {
            int fs = fetch % STAGES;
            load_tile_pad<BM>(sA0 + fs * PAD_STAGE_A_BYTES, A, m0, M, fetch * BK, K, tid);
            load_tile_pad<BN>(sB0 + fs * PAD_STAGE_B_BYTES, B, n0, N, fetch * BK, K, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);

        unsigned aStage = sA0 + cur * PAD_STAGE_A_BYTES;
        unsigned bStage = sB0 + cur * PAD_STAGE_B_BYTES;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 32)
        {
            unsigned bfrag[8][2];
            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                unsigned base = bStage + (warpN * 64 + j * 8 + (lane & 7)) * PAD_STRIDE + kk + ((lane >> 3) & 1) * 16;
                ldmatrix_x2(base, bfrag[j][0], bfrag[j][1]);
            }
            unsigned afrag[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                unsigned base = aStage + (warpM * 64 + i * 16 + (lane & 15)) * PAD_STRIDE + kk + (lane >> 4) * 16;
                ldmatrix_x4(base, afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                unsigned a0 = afrag[i][0], a1 = afrag[i][1], a2 = afrag[i][2], a3 = afrag[i][3];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
                }
            }
        }
    }

    __half* slab = (__half*)smem;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int half = 0; half < 2; half++)
        {
            int row = m0 + warpM * 64 + i * 16 + gid + half * 8;
            float as = row < (int)M ? actScale[row] : 0.0f;
            int slabRow = warpM * 8 + gid;
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                int cs = warpN * 64 + j * 8 + tig * 2;
                int col = n0 + cs;
                float t0 = (float)acc[i][j][half * 2 + 0] * as;
                float t1 = (float)acc[i][j][half * 2 + 1] * as;
                float v0 = bias ? fmaf(t0, wScale[col], bias[col]) : t0 * wScale[col];
                float v1 = bias ? fmaf(t1, wScale[col + 1], bias[col + 1]) : t1 * wScale[col + 1];
                if (actMode == 1u) { v0 = gelu_tanh(v0); v1 = gelu_tanh(v1); }
                *(__half2*)(slab + slabRow * BN + cs) = __halves2half2(__float2half(v0), __float2half(v1));
            }
            __syncthreads();
            #pragma unroll
            for (int p = 0; p < 2; p++)
            {
                int slot = tid + p * 256;
                int wr = slot >> 5;
                int wc = (slot & 31) * 8;
                int outRow = m0 + (wr >> 3) * 64 + i * 16 + half * 8 + (wr & 7);
                if (outRow < (int)M)
                    *(int4*)(D + (size_t)outRow * N + n0 + wc) = *(const int4*)(slab + wr * BN + wc);
            }
        }
    }
}
