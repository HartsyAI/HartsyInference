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
// Tiling: block 128(M) x 128(N) x 64(K), 8 warps as 2(M) x 4(N), warp tile 64 x 32 = 4x4 mma m16n8k32 tiles
// (64 int32 accumulator registers/thread). Two shared-memory stages, filled with cp.async.
//
// Shared row stride is padded to BK+16: at the natural 64-byte stride the 8 rows a warp reads for one A
// fragment land on only 8 distinct banks (row*16+tig mod 32 repeats every 2 rows), a 4-way conflict on every
// fragment load. 80 bytes makes the 8 rows hit 8 distinct bank groups, and keeps the 16-byte cp.async
// alignment.

#include <cuda_fp16.h>

#define BM 128
#define BN 128
#define BK 64
#define SMEM_STRIDE (BK + 16)          // 80 B, see note above
#define STAGE_BYTES (BM * SMEM_STRIDE) // per stage, per operand

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

// Stages one 128x64 int8 tile of a [rows, K] row-major matrix into shared. 128 rows x 4 chunks of 16 B,
// 256 threads => 2 chunks each.
__device__ __forceinline__ void load_tile(unsigned sBase, const signed char* g, int row0, int rows,
                                          int k0, int K, int tid)
{
    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        int idx = tid + i * 256;
        int r = idx >> 2;                 // 0..127
        int c = (idx & 3) * 16;           // 0,16,32,48
        int gr = row0 + r;
        cp_async16(sBase + r * SMEM_STRIDE + c, g + (size_t)gr * K + k0 + c, gr < rows);
    }
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
    unsigned sB0 = sA0 + 2 * STAGE_BYTES;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;      // mma fragment addressing
    const int warpM = warp >> 2, warpN = warp & 3;  // 2 x 4 warps

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0;

    const int kTiles = K / BK;

    load_tile(sA0, A, m0, M, 0, K, tid);
    load_tile(sB0, B, n0, N, 0, K, tid);
    asm volatile("cp.async.commit_group;\n" ::);

    for (int kt = 0; kt < kTiles; kt++)
    {
        int cur = kt & 1, nxt = cur ^ 1;
        if (kt + 1 < kTiles)
        {
            load_tile(sA0 + nxt * STAGE_BYTES, A, m0, M, (kt + 1) * BK, K, tid);
            load_tile(sB0 + nxt * STAGE_BYTES, B, n0, N, (kt + 1) * BK, K, tid);
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        unsigned aStage = sA0 + cur * STAGE_BYTES;
        unsigned bStage = sB0 + cur * STAGE_BYTES;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 32)
        {
            unsigned bfrag[4][2];
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                unsigned base = bStage + (warpN * 32 + j * 8 + gid) * SMEM_STRIDE + kk + tig * 4;
                asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(bfrag[j][0]) : "r"(base));
                asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(bfrag[j][1]) : "r"(base + 16));
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                unsigned rowA = aStage + (warpM * 64 + i * 16 + gid) * SMEM_STRIDE + kk + tig * 4;
                unsigned rowB = rowA + 8 * SMEM_STRIDE;
                unsigned a0, a1, a2, a3;
                asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a0) : "r"(rowA));
                asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a1) : "r"(rowB));
                asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a2) : "r"(rowA + 16));
                asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a3) : "r"(rowB + 16));
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
                }
            }
        }
        __syncthreads();
    }

    // ── Fused dequant epilogue: the int32 accumulator never leaves registers ──
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
        {
            #pragma unroll
            for (int half = 0; half < 2; half++)
            {
                int row = m0 + warpM * 64 + i * 16 + gid + half * 8;
                if (row >= (int)M) continue;
                float as = actScale[row];
                #pragma unroll
                for (int t = 0; t < 2; t++)
                {
                    int col = n0 + warpN * 32 + j * 8 + tig * 2 + t;
                    if (col >= (int)N) continue;
                    float v = (float)acc[i][j][half * 2 + t] * as * wScale[col];
                    if (bias) v += bias[col];
                    if (actMode == 1u) v = gelu_tanh(v);
                    D[(size_t)row * N + col] = __float2half(v);
                }
            }
        }
    }
}
