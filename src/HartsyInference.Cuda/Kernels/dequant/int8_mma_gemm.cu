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
// Measured on a 4090 against cuBLASLt-GEMM + w8a8_dequant_bias, bit-exact throughout, COLD L2 (see ColdBuffers
// in Int8MmaGemmTests — reusing one resident weight flatters this kernel and inverted a conclusion once):
// attn_qkvo 4992x4096x4096 413 TOPS = +6.1% vs the pair, ffn_up 4992x16384x4096 -7.0%, ffn_down
// 4992x4096x16384 -28%. CudaBackend.UseFusedMmaGemm admits only the winning regime.
//
// WHY 128x256 AND NOT 128x128 — arithmetic intensity, and it was the single biggest win (288 -> 341 TOPS).
// A 128x128 block reads 16 KB of operands per k-tile to do 2.1 MFLOP — 131 flop/byte — so sustaining the 4090's
// 660 TOPS would demand ~5 TB/s out of L2, at its ceiling and far past HBM. 128x256 reads 24 KB for 4.2 MFLOP,
// and per unit of OUTPUT the A tile is reused twice as much. This is comfy-kitchen's shipped tile for these
// shapes. The cost is that 128 accumulator registers (195 total, no spill) force one block per SM.
//
// THE LIMITER IS L2 BANDWIDTH — measured with Nsight Compute, not inferred. At attn_qkvo:
//   Memory (L2) throughput 85.2%, Compute (SM) throughput 57.6%, DRAM only 17.9%, zero spilling.
// So this kernel goes faster only by moving FEWER BYTES THROUGH L2. Instruction mix, occupancy and pipeline
// depth are all secondary, which is why every attempt at them landed in the 1-6% band.
//
// AND THE WASTE IS LOCALISED: global STORES average 32.0 of 32 bytes per sector (optimal, after the epilogue
// fix below), while global LOADS average **21.33 of 32** — exactly 32 x 2/3, the signature of a 64-byte run
// landing across THREE 32-byte sectors instead of two. Those loads are the cp.async operand fetches, they are
// 97.5% L2 hits, and they are the traffic saturating the bottleneck. Ideal operand traffic is 981 MB against
// 47.4M sectors measured versus 31.9M ideal — 491 MB of pure waste, and that waste IS the remaining gap to
// cuBLASLt (removing it at constant bandwidth lands ~620 TOPS against its 569).
//
// The run length per row is exactly BK bytes, so with BK=64 a row contributes only 64 contiguous bytes before
// jumping K. NEXT EXPERIMENT (untested, and this file's history says test before believing): BK=128 with an
// XOR-swizzled unpadded shared layout — 128-byte runs, and (128+256)*128*2 stages = 98,304 B still fits the
// 99 KB opt-in, where the padded BK=128 layout does not. It would also fix the 8-way shared-store bank conflict
// ncu reports, though that one is NOT on the critical path (L1/TEX is only 32-34% busy against L2's 85%).
//
// Occupancy is 16.7% (8 warps, one block/SM) and ncu flags an "83% local speedup" against it — IGNORE THAT
// for this kernel. The 128 accumulator registers that force one block per SM are the same thing that buys the
// arithmetic intensity; raising occupancy means a smaller tile, which raises L2 traffic, which is the actual
// constraint. ncu's occupancy advice does not know that.
//
// RULED OUT as limiters, each measured, so do not re-chase:
//   - Shared-load instruction count. ldmatrix cut 24 fragment loads per k-step to 8 and bought 1-6%.
//   - Occupancy per the paragraph above.
//   - Shared-memory bank conflicts. ncu reports 45% excessive shared wavefronts, and fixing them is NOT the
//     win it looks like: L1/TEX throughput is only 32.9% against L2's 85.2%, so the shared path has headroom
//     to spare. An earlier version of this comment claimed the 80-byte stride was "conflict-free" and sized an
//     XOR swizzle at "low single digits" — the first half was wrong (rows 0 and 8 do collide) and the second
//     half is right, but for a reason that only a profiler could establish.
//
// The epilogue, by contrast, was worth ~20% and was nearly skipped on the reasoning that "cuBLASLt writes twice
// these bytes as int32 and is still faster". That compares bytes moved; the right metric is SECTORS TOUCHED —
// its int32 store is perfectly coalesced and mma's fragment layout is not. See the epilogue for the fix.
//
// STAGES > 2 needs more than the 48 KB default of dynamic shared, so the launcher must opt in via
// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES — and must ask for EXACTLY what is used, never the SM ceiling,
// or the driver sizes occupancy against the ceiling and silently drops a block per SM. Keep STAGES and
// CudaKernels.Int8MmaSharedBytes in step.
//
// Shared row stride is padded to BK+16: at the natural 64-byte stride the 8 rows a warp reads for one A
// fragment land on only 8 distinct banks (row*16+tig mod 32 repeats every 2 rows), a 4-way conflict on every
// fragment load. 80 bytes makes the 8 rows hit 8 distinct bank groups, and keeps the 16-byte cp.async
// alignment. It is not perfect — ldmatrix rows 0 and 8 still share a bank — but among 16-byte-aligned strides
// (64/80/96/112/128) 80 already gives the longest conflict-free run; going further needs an XOR swizzle, worth
// low single digits given ldmatrix itself bought 1-6%.

#include <cuda_fp16.h>

#define BM 128
#define BN 256
#define BK 64
#define STAGES 3
#define SMEM_STRIDE (BK + 16)             // 80 B, see note above
#define STAGE_A_BYTES (BM * SMEM_STRIDE)  // A and B stages differ in size now that BN != BM
#define STAGE_B_BYTES (BN * SMEM_STRIDE)

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

// Stages one TILE_ROWS x 64 int8 tile of a [rows, K] row-major matrix into shared. Each row is 4 chunks of
// 16 B, so 256 threads cover TILE_ROWS*4/256 chunks each — 2 for the 128-row A tile, 4 for the 256-row B tile.
template <int TILE_ROWS>
__device__ __forceinline__ void load_tile(unsigned sBase, const signed char* g, int row0, int rows,
                                          int k0, int K, int tid)
{
    #pragma unroll
    for (int i = 0; i < TILE_ROWS / 64; i++)
    {
        int idx = tid + i * 256;
        int r = idx >> 2;                 // 0..TILE_ROWS-1
        int c = (idx & 3) * 16;           // 0,16,32,48
        int gr = row0 + r;
        cp_async16(sBase + r * SMEM_STRIDE + c, g + (size_t)gr * K + k0 + c, gr < rows);
    }
}

// ── ldmatrix fragment loads ────────────────────────────────────────────────
// One `ldmatrix.x4` replaces the 4 scalar ld.shared.b32 an A fragment needed, and one `.x2` the 2 a B fragment
// needed: 24 shared loads per k-step become 8. That ratio is the mainloop's problem — 48 ld.shared against 32
// mma per k-tile leaves the tensor cores waiting on the LSU.
//
// ldmatrix moves 8x8 tiles of 16-bit elements, i.e. 8 rows x 16 BYTES, which is exactly how m16n8k32 wants its
// int8 operands laid out, so no repack is needed — only the right per-lane address. Lane L addresses row L%8 of
// matrix L/8; mapping that onto mma's operand layout gives:
//   A (16 rows x 32 int8): matrices are (rows 0-7, bytes 0-15), (rows 8-15, bytes 0-15), (rows 0-7, bytes
//   16-31), (rows 8-15, bytes 16-31) -> addr = base + (L%16)*stride + (L/16)*16, and {r0..r3} land directly in
//   mma's {a0..a3}.
//   B (8 rows x 32 int8): two matrices, bytes 0-15 then 16-31 -> addr = base + (L%8)*stride + ((L/8)&1)*16.
// The existing 80-byte row stride is already conflict-free for this access: 8 rows step 20 banks apart, which
// visits 8 distinct banks before repeating.
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
    unsigned sB0 = sA0 + STAGES * STAGE_A_BYTES;

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
            load_tile<BM>(sA0 + s * STAGE_A_BYTES, A, m0, M, s * BK, K, tid);
            load_tile<BN>(sB0 + s * STAGE_B_BYTES, B, n0, N, s * BK, K, tid);
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
            load_tile<BM>(sA0 + fs * STAGE_A_BYTES, A, m0, M, fetch * BK, K, tid);
            load_tile<BN>(sB0 + fs * STAGE_B_BYTES, B, n0, N, fetch * BK, K, tid);
        }
        // Committed unconditionally: the wait above counts groups, so a skipped tail fetch still needs one.
        asm volatile("cp.async.commit_group;\n" ::);

        unsigned aStage = sA0 + cur * STAGE_A_BYTES;
        unsigned bStage = sB0 + cur * STAGE_B_BYTES;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 32)
        {
            unsigned bfrag[8][2];
            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                unsigned base = bStage + (warpN * 64 + j * 8 + (lane & 7)) * SMEM_STRIDE + kk + ((lane >> 3) & 1) * 16;
                ldmatrix_x2(base, bfrag[j][0], bfrag[j][1]);
            }
            unsigned afrag[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                unsigned base = aStage + (warpM * 64 + i * 16 + (lane & 15)) * SMEM_STRIDE + kk + (lane >> 4) * 16;
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
        // No trailing barrier: the next iteration's top-of-loop __syncthreads() is what protects this stage
        // from the prefetch that overwrites it, and a second barrier here would only serialize the mainloop.
    }

    // ── Fused dequant epilogue: the int32 accumulator never leaves registers ──
    // THE STORE PATTERN IS WORTH REAL TIME HERE — measured, against expectation. Packing the two results a thread
    // holds into one 4-byte __half2 instead of two 2-byte stores took attn_qkvo 343 -> 390 TOPS and ffn_up
    // 331 -> 376. The reasoning that talked me out of trying it first ("cuBLASLt writes twice these bytes as int32
    // and is still faster, so stores cannot be the gap") was wrong: its int32 store is perfectly coalesced while
    // this fragment layout scatters, so bytes moved is the wrong comparison — sectors touched is the right one.
    //
    // mma leaves each thread holding COLUMN PAIRS (tig*2, tig*2+1) of a row, adjacent in the row-major output.
    // N is a multiple of the 256-wide block tile, so a pair never straddles the edge and needs no per-element
    // bound check; only M is predicated. Still not ideal: a warp covers 8 rows x 16 contiguous bytes per store,
    // half of a 32-byte sector. Staging the tile through shared to emit fully coalesced rows is the next step.
    // Staged through shared one 16-row x 256-col slab at a time (8 KB, reusing the mainloop's stage buffer, which
    // is dead by now): threads scatter their fragment-shaped pieces into the slab, then the block re-reads it and
    // writes 16 contiguous BYTES per lane, so a warp emits 512 contiguous bytes per instruction instead of eight
    // 16-byte pieces. The 16 rows of a slab are two groups of 8 (warpM splits them 64 apart), so the slab is
    // indexed [warpM*8 + gid] and the writer re-derives the real row from that.
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
            // int4s per lane. Both move the same bytes; only this one is coalesced. With 16 lanes per row each
            // writing 32 B as two 16-B stores, every store instruction strides 32 B and touches each 32-byte
            // sector half-used — ncu measured 16.6M excessive sectors, 34% of all global sectors, on a kernel
            // that is L2-bandwidth bound at 84%. Here a warp's 32 lanes cover one whole 512-byte output row in
            // a single instruction.
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
