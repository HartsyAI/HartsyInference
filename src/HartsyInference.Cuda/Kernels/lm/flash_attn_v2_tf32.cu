// FlashAttention-2, TF32 tensor cores, F32 accumulate. Fused query-tiled online-softmax attention that NEVER
// materializes the [Sq x Skv] score matrix (the profiled #1 GPU cost in the video DiT). No-mask MHA only,
// head_dim D in {64,128}, no GQA (Hkv==Hq). Other cases keep the materialized / naive paths in CudaBackend.
//
// TF32 (not F16) inputs: keeps the full F32 exponent range, so pre-softmax scores can't overflow even for
// unbounded-score archs (fp8) — this is the safety fix for the F16 blackout. All softmax state (m,l,O,S) is F32.
//
// Layout: Q,out [B, Hq, Sq, D] row-major; K,V [B, Hq, Skv, D] row-major. One block per (batch, head, query-tile).
// Grid = (ceil(Sq/BR), Hq, B). Block = 128 threads (4 warps). BR=64 query rows, BC=32 key cols per step.
//
// Build (no nvcc on this box — use nvrtc):
//   TINC=".../triton/backends/nvidia/include"
//   LD_LIBRARY_PATH=~/.local/lib/cuda13 ../nvrtc_compile flash_attn_v2_tf32.cu flash_attn_v2_tf32.ptx compute_80 "$TINC"
//   cp flash_attn_v2_tf32.ptx ../../../src/HartsyInference.Cuda/Ptx/
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// M2 occupancy: BR=32/BC=16 (D=128) → smem = K(8KB)+V(8KB)+S(2KB)+O(16KB) = 34KB < 48KB default → 2 blocks/SM
// (vs 72KB/1-block at BR=64/BC=32). BR=32 = 2 warps × 16 rows; block = 64 threads.
#define BR 32          // query rows per block
#define BC 16          // key/value columns per K/V step
#define WARPS 2        // 64 threads
#define NEG_INF (-3.402823466e+38f)

// WMMA tile: 16x16x8 TF32. BR=64 = 4 row-tiles (one per warp). BC=32 = 2 col-tiles. D contracted in steps of 8.
// Each warp owns 16 query rows (rows [warp*16, warp*16+16)). For S = Q@K^T [16 x BC] it accumulates over D.
// For O += P@V [16 x D] it accumulates over BC.

extern "C" __global__ void lm_flash_attn_v2_tf32(
    float* __restrict__ out,          // [B, Hq, Sq, D]
    const float* __restrict__ Q,      // [B, Hq, Sq, D]
    const float* __restrict__ K,      // [B, Hq, Skv, D]
    const float* __restrict__ V,      // [B, Hq, Skv, D]
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv, unsigned int D,
    float scale)
{
    const unsigned int qtile = blockIdx.x;   // which BR-row query tile
    const unsigned int h     = blockIdx.y;
    const unsigned int b     = blockIdx.z;
    const unsigned int warp  = threadIdx.x >> 5;   // 0..3
    const unsigned int lane  = threadIdx.x & 31;

    const unsigned int q0 = qtile * BR;            // first query row of this tile
    if (q0 >= Sq) return;

    // Base pointers for this (b,h).
    const size_t qkvHeadStrideQ = (size_t)Sq * D;
    const size_t qkvHeadStrideKV = (size_t)Skv * D;
    const float* Qh = Q + ((size_t)b * Hq + h) * qkvHeadStrideQ;
    const float* Kh = K + ((size_t)b * Hq + h) * qkvHeadStrideKV;
    const float* Vh = V + ((size_t)b * Hq + h) * qkvHeadStrideKV;
    float* Oh = out + ((size_t)b * Hq + h) * qkvHeadStrideQ;

    // Shared memory: K tile [BC][D], V tile [BC][D], and S scratch [BR][BC] for the softmax round-trip.
    extern __shared__ float smem[];
    float* Ksh = smem;                       // BC*D
    float* Vsh = Ksh + (size_t)BC * D;       // BC*D
    float* Ssh = Vsh + (size_t)BC * D;       // BR*BC

    // This warp owns 16 query rows: rows [warp*16 .. warp*16+16) within the tile.
    const unsigned int rowBase = warp * 16;

    // Online-softmax state per query row, kept in F32 registers. 16 rows/warp, one lane-strided owner:
    // lane l owns rows where (row % 32)... simpler: keep m/l per-row in shared? For clarity use per-warp smem arrays.
    __shared__ float m_state[BR];    // running max
    __shared__ float l_state[BR];    // running sum
    for (unsigned int i = threadIdx.x; i < BR; i += blockDim.x) { m_state[i] = NEG_INF; l_state[i] = 0.0f; }

    // O accumulator [BR][D] in shared memory (BR*D floats). Init 0.
    // NOTE: BR*D = 64*128 = 8192 floats = 32KB — with K/V/S tiles this exceeds 48KB, so O lives in GLOBAL out and we
    // accumulate there via read-modify-write per K/V step is too slow. Instead: keep O in shared, and shrink other
    // tiles. Budget (D=128): K 16KB + V 16KB + S 8KB + O 32KB = 72KB -> needs cuFuncSetAttribute opt-in (99KB avail).
    float* Osh = Ssh + (size_t)BR * BC;      // BR*D
    for (unsigned int i = threadIdx.x; i < (unsigned int)BR * D; i += blockDim.x) Osh[i] = 0.0f;
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> qf;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> kf;   // K as [D x BC] col-major = K[BC x D] row-major transposed
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> sf;

    const unsigned int nKsteps = (Skv + BC - 1) / BC;
    for (unsigned int kc = 0; kc < nKsteps; kc++)
    {
        const unsigned int c0 = kc * BC;
        const unsigned int curBC = min((unsigned int)BC, Skv - c0);

        // Stage K,V [curBC x D] into shared (coalesced).
        for (unsigned int i = threadIdx.x; i < curBC * D; i += blockDim.x)
        {
            Ksh[i] = Kh[(size_t)(c0) * D + i];
            Vsh[i] = Vh[(size_t)(c0) * D + i];
        }
        __syncthreads();

        // S[16 x BC] = scale * Q[16 x D] @ K[BC x D]^T, per warp (its 16 rows). BC=32 = two 16-col fragments.
        for (unsigned int ct = 0; ct < BC; ct += 16)
        {
            wmma::fill_fragment(sf, 0.0f);
            for (unsigned int kk = 0; kk < D; kk += 8)
            {
                // Q fragment: rows [q0+rowBase .. +16), cols [kk..kk+8)
                wmma::load_matrix_sync(qf, Qh + (size_t)(q0 + rowBase) * D + kk, D);
                // K fragment as col-major [D x 16]: Ksh is [BC x D] row-major; the 16 keys [ct..ct+16) each length D.
                // col_major matrix_b with leading dim D loads Ksh[key][kk..] as a column -> gives K^T. Load from
                // Ksh + ct*D + kk with ld=D.
                wmma::load_matrix_sync(kf, Ksh + (size_t)ct * D + kk, D);
                for (int i = 0; i < qf.num_elements; i++) qf.x[i] = wmma::__float_to_tf32(qf.x[i]);
                for (int i = 0; i < kf.num_elements; i++) kf.x[i] = wmma::__float_to_tf32(kf.x[i]);
                wmma::mma_sync(sf, qf, kf, sf);
            }
            // store S tile [16 x 16] to Ssh[rowBase..][ct..]
            wmma::store_matrix_sync(Ssh + (size_t)rowBase * BC + ct, sf, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax over this BC block, per query row (one thread per row for the reduction).
        for (unsigned int r = threadIdx.x; r < BR; r += blockDim.x)
        {
            if (q0 + r >= Sq) continue;
            float* Srow = Ssh + (size_t)r * BC;
            float rowMax = NEG_INF;
            for (unsigned int c = 0; c < curBC; c++) { float s = Srow[c] * scale; Srow[c] = s; if (s > rowMax) rowMax = s; }
            float mPrev = m_state[r];
            float mNew = fmaxf(mPrev, rowMax);
            float corr = __expf(mPrev - mNew);
            float rowSum = 0.0f;
            for (unsigned int c = 0; c < curBC; c++) { float p = __expf(Srow[c] - mNew); Srow[c] = p; rowSum += p; }
            for (unsigned int c = curBC; c < BC; c++) Srow[c] = 0.0f;   // pad
            l_state[r] = l_state[r] * corr + rowSum;
            m_state[r] = mNew;
            // rescale existing O row by corr
            float* Orow = Osh + (size_t)r * D;
            for (unsigned int d = 0; d < D; d++) Orow[d] *= corr;
        }
        __syncthreads();

        // O[16 x D] += P[16 x BC] @ V[BC x D], per warp. P = Ssh (F32, TF32-cast); accumulate into Osh.
        for (unsigned int dt = 0; dt < D; dt += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> pf;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> vf;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> of;
            // load current O tile [16 x 16] from Osh to accumulate
            wmma::load_matrix_sync(of, Osh + (size_t)rowBase * D + dt, D, wmma::mem_row_major);
            for (unsigned int cc = 0; cc < BC; cc += 8)
            {
                wmma::load_matrix_sync(pf, Ssh + (size_t)rowBase * BC + cc, BC);
                wmma::load_matrix_sync(vf, Vsh + (size_t)cc * D + dt, D);
                for (int i = 0; i < pf.num_elements; i++) pf.x[i] = wmma::__float_to_tf32(pf.x[i]);
                for (int i = 0; i < vf.num_elements; i++) vf.x[i] = wmma::__float_to_tf32(vf.x[i]);
                wmma::mma_sync(of, pf, vf, of);
            }
            wmma::store_matrix_sync(Osh + (size_t)rowBase * D + dt, of, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Epilogue: O_row /= l, write to global.
    for (unsigned int r = threadIdx.x; r < BR; r += blockDim.x)
    {
        if (q0 + r >= Sq) continue;
        float inv = 1.0f / l_state[r];
        float* Orow = Osh + (size_t)r * D;
        float* Og = Oh + (size_t)(q0 + r) * D;
        for (unsigned int d = 0; d < D; d++) Og[d] = Orow[d] * inv;
    }
}
