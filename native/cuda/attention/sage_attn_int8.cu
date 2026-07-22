// SageAttention-v1 INT8 flash attention (arXiv:2410.02367 recipe) for diffusion DiTs — INT8 QK^T on the
// IMMA tensor cores with K channel-mean smoothing, online softmax + PV kept in F32/TF32. Non-causal,
// no-mask, fixed-seqlen (the diffusion simplification, DEEP_KERNEL_OPTIMIZATION §2), head_dim D ∈ {64,128}.
//
// Why INT8: SM 8.6 runs s8×s8→s32 IMMA at 2× the FP16 tensor-core rate (4× TF32), and QK^T is the larger
// GEMM of attention at video seqlens. Why K-smoothing: DiT K activations carry channel-consistent outliers;
// per-row INT8 quant collapses on them, and subtracting K's per-channel mean changes every logit in a row
// by the same constant → softmax-invariant (validated numerically by SageAttentionReferenceTests, the
// correctness oracle for this kernel). PV deliberately stays TF32 (flash_attn_v2_tf32 structure) instead of
// the paper's F16: correctness-first v0 — QK^T is where the tensor-core win is; revisit PV dtype only if
// the ncu profile says PV dominates.
//
// Pipeline per forward (all launched by CudaBackend.SageAttentionInt8, gated on HARTSY_SAGE_ATTN=1):
//   1. sage_k_mean_f32        — per-(b,h) channel mean of K over the sequence.
//   2. sage_quant_q_int8_f32  — per-row absmax INT8 quant of Q; attn scale folded into the row scale.
//   3. sage_quant_k_int8_f32  — per-row absmax INT8 quant of (K − mean).
//   4. sage_attn_int8_f32     — fused flash loop: S = dequant(Q8 @ K8^T), online softmax, O += P @ V.
//
// Layout: Q,out [B, Hq, Sq, D] row-major; K,V [B, Hq, Skv, D] row-major; no GQA (Hkv==Hq). Q8/K8 mirror
// their sources as int8; qscale/kscale are [B, Hq, S] F32. Grid/block per kernel documented at each entry.
// Sq must be a multiple of 32 (BR) — the WMMA Q-fragment loads are unguarded, same as flash_attn_v2_tf32;
// the C# dispatch shape-gates this (diffusion img streams are 16²-multiples). Skv tail is handled (curBC).
//
// Build: nvcc -ptx -arch=sm_80 sage_attn_int8.cu -o sage_attn_int8.ptx   (see build.sh; PTX must say
// .version 9.0 — pinned toolchain ~/.local/cuda-tools-13.0, see INFERENCE_ACCEL_GRIND §H1.1 trap note.)
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define BR 32          // query rows per block (2 warps × 16 WMMA rows)
#define BC 16          // key/value columns per K/V step
#define WARPS 2        // 64 threads
#define NEG_INF (-3.402823466e+38f)

// ---------------------------------------------------------------------------------------------------------
// 1. K per-channel mean over the sequence, per (b,h): kmean[b,h,d] = (1/Skv) Σ_s K[b,h,s,d].
//    Grid = (Hq, B), block = 128 threads; thread d owns channel d (guarded d < D), strides the sequence.
//    Consecutive threads read consecutive addresses per row → fully coalesced.
extern "C" __global__ void sage_k_mean_f32(
    float* __restrict__ kmean,        // [B, Hq, D]
    const float* __restrict__ K,      // [B, Hq, Skv, D]
    unsigned int B, unsigned int Hq, unsigned int Skv, unsigned int D)
{
    const unsigned int h = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (d >= D) return;

    const float* Kh = K + ((size_t)b * Hq + h) * ((size_t)Skv * D);
    float acc = 0.0f;
    for (unsigned int s = 0; s < Skv; s++) acc += Kh[(size_t)s * D + d];
    kmean[((size_t)b * Hq + h) * D + d] = acc / (float)Skv;
}

// ---------------------------------------------------------------------------------------------------------
// Per-row absmax INT8 quant, one warp per row (quantize_activation_q8_1_f32 pattern): each lane strides the
// row, warp-reduces amax via __shfl_xor_sync, then writes round-nearest int8 clamped to [-127,127].
// smoothSub: optional [B,Hq,D] per-channel vector subtracted before quantization (K-smoothing); null for Q.
// scaleMul: folded into the stored row scale (attn softmax scale for Q, 1.0 for K) so the flash kernel's
// dequant is one qscale[r]*kscale[c] multiply per score.
__device__ __forceinline__ void sage_quant_row(
    signed char* __restrict__ out8, float* __restrict__ scales,
    const float* __restrict__ src, const float* __restrict__ smoothSub,
    unsigned int rows, unsigned int Hq, unsigned int S, unsigned int D, float scaleMul,
    unsigned int warpsPerBlock)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = blockIdx.x * warpsPerBlock + warp;   // global row index over B*Hq*S
    if (row >= rows) return;

    const float* srcRow = src + (size_t)row * D;
    // (b,h) for the smoothing vector: rows are laid out [B, Hq, S].
    const float* sub = nullptr;
    if (smoothSub != nullptr)
    {
        const unsigned int bh = row / S;              // b*Hq + h
        sub = smoothSub + (size_t)bh * D;
    }

    float amax = 0.0f;
    for (unsigned int d = lane; d < D; d += 32)
    {
        float v = srcRow[d];
        if (sub != nullptr) v -= sub[d];
        amax = fmaxf(amax, fabsf(v));
    }
    for (unsigned int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));

    const float qscale = amax > 0.0f ? amax / 127.0f : 1.0f;      // all-zero row → zeros at scale 1
    const float inv = 1.0f / qscale;
    signed char* outRow = out8 + (size_t)row * D;
    for (unsigned int d = lane; d < D; d += 32)
    {
        float v = srcRow[d];
        if (sub != nullptr) v -= sub[d];
        int q = __float2int_rn(v * inv);
        q = max(-127, min(127, q));
        outRow[d] = (signed char)q;
    }
    if (lane == 0) scales[row] = qscale * scaleMul;
}

// Grid = ceil(B*Hq*S / 4), block = 128 (4 warps → 4 rows/block).
extern "C" __global__ void sage_quant_q_int8_f32(
    signed char* __restrict__ q8,     // [B, Hq, Sq, D]
    float* __restrict__ qscale,       // [B, Hq, Sq] (attn scale folded in)
    const float* __restrict__ Q,      // [B, Hq, Sq, D]
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int D, float attnScale)
{
    sage_quant_row(q8, qscale, Q, nullptr, B * Hq * Sq, Hq, Sq, D, attnScale, blockDim.x >> 5);
}

extern "C" __global__ void sage_quant_k_int8_f32(
    signed char* __restrict__ k8,     // [B, Hq, Skv, D]
    float* __restrict__ kscale,       // [B, Hq, Skv]
    const float* __restrict__ K,      // [B, Hq, Skv, D]
    const float* __restrict__ kmean,  // [B, Hq, D]
    unsigned int B, unsigned int Hq, unsigned int Skv, unsigned int D)
{
    sage_quant_row(k8, kscale, K, kmean, B * Hq * Skv, Hq, Skv, D, 1.0f, blockDim.x >> 5);
}

// ---------------------------------------------------------------------------------------------------------
// 4. Fused flash loop. Structure = flash_attn_v2_tf32 with the QK^T stage swapped to s8 IMMA
//    (m16n16k16 s8·s8→s32: D=128 contracts in 8 mma steps vs TF32's 16, at 2× the issue rate).
//    Dequant (qscale[r] * kscale[c], attn scale pre-folded into qscale) happens inside the per-row
//    softmax pass, which already touches every score — zero extra passes.
//
// Grid = (ceil(Sq/BR), Hq, B), block = 64 threads (2 warps; warp w owns query rows [w*16, w*16+16)).
// Dynamic SMEM layout (f32 arrays first, all sizes 32B-multiples; K8 tile last):
//   Vsh [BC*D] f32 | Ssh [BR*BC] f32/s32 dual-use | Osh [BR*D] f32 | K8sh [BC*D] s8
// D=128: 8KB + 2KB + 16KB + 2KB = 28KB < 48KB default → 2 blocks/SM, no opt-in attribute needed.
extern "C" __global__ void sage_attn_int8_f32(
    float* __restrict__ out,              // [B, Hq, Sq, D]
    const signed char* __restrict__ Q8,   // [B, Hq, Sq, D]
    const float* __restrict__ qscale,     // [B, Hq, Sq]
    const signed char* __restrict__ K8,   // [B, Hq, Skv, D]
    const float* __restrict__ kscale,     // [B, Hq, Skv]
    const float* __restrict__ V,          // [B, Hq, Skv, D]
    unsigned int B, unsigned int Hq, unsigned int Sq, unsigned int Skv, unsigned int D)
{
    const unsigned int qtile = blockIdx.x;
    const unsigned int h     = blockIdx.y;
    const unsigned int b     = blockIdx.z;
    const unsigned int warp  = threadIdx.x >> 5;
    const unsigned int q0    = qtile * BR;
    if (q0 >= Sq) return;

    const size_t headQ  = (size_t)Sq * D;
    const size_t headKV = (size_t)Skv * D;
    const signed char* Q8h = Q8 + ((size_t)b * Hq + h) * headQ;
    const signed char* K8h = K8 + ((size_t)b * Hq + h) * headKV;
    const float* Vh  = V   + ((size_t)b * Hq + h) * headKV;
    const float* qsh = qscale + ((size_t)b * Hq + h) * Sq;
    const float* ksh = kscale + ((size_t)b * Hq + h) * Skv;
    float* Oh = out + ((size_t)b * Hq + h) * headQ;

    extern __shared__ float smem[];
    float* Vsh = smem;                                   // BC*D f32
    float* Ssh = Vsh + (size_t)BC * D;                   // BR*BC f32 (s32 during dequant window)
    float* Osh = Ssh + (size_t)BR * BC;                  // BR*D f32
    signed char* K8sh = (signed char*)(Osh + (size_t)BR * D);   // BC*D s8
    int* Ssh32 = (int*)Ssh;

    const unsigned int rowBase = warp * 16;

    __shared__ float m_state[BR];
    __shared__ float l_state[BR];
    __shared__ float kscale_sh[BC];
    for (unsigned int i = threadIdx.x; i < BR; i += blockDim.x) { m_state[i] = NEG_INF; l_state[i] = 0.0f; }
    for (unsigned int i = threadIdx.x; i < (unsigned int)BR * D; i += blockDim.x) Osh[i] = 0.0f;
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> qf;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> kf;   // K8sh [BC x D] row-major read col-major = K^T
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> sf;

    const unsigned int nKsteps = (Skv + BC - 1) / BC;
    for (unsigned int kc = 0; kc < nKsteps; kc++)
    {
        const unsigned int c0 = kc * BC;
        const unsigned int curBC = min((unsigned int)BC, Skv - c0);

        // Stage K8 (bytes), V (f32) and this step's kscales; zero-pad the K8 tail rows so the padded
        // columns produce deterministic (garbage-free) scores — they're discarded after softmax anyway.
        for (unsigned int i = threadIdx.x; i < (unsigned int)BC * D; i += blockDim.x)
        {
            K8sh[i] = (i < curBC * D) ? K8h[(size_t)c0 * D + i] : (signed char)0;
            Vsh[i]  = (i < curBC * D) ? Vh[(size_t)c0 * D + i] : 0.0f;
        }
        for (unsigned int i = threadIdx.x; i < BC; i += blockDim.x)
            kscale_sh[i] = (i < curBC) ? ksh[c0 + i] : 0.0f;
        __syncthreads();

        // S_s32[16 x BC] = Q8[16 x D] @ K8^T, per warp. BC=16 = one col tile; D contracted in k=16 steps.
        wmma::fill_fragment(sf, 0);
        for (unsigned int kk = 0; kk < D; kk += 16)
        {
            wmma::load_matrix_sync(qf, Q8h + (size_t)(q0 + rowBase) * D + kk, D);
            wmma::load_matrix_sync(kf, K8sh + (size_t)0 * D + kk, D);
            wmma::mma_sync(sf, qf, kf, sf);
        }
        wmma::store_matrix_sync(Ssh32 + (size_t)rowBase * BC, sf, BC, wmma::mem_row_major);
        __syncthreads();

        // Dequant + online softmax per query row (one thread per row). Reads s32 in-place, writes back the
        // F32 probability — same buffer, same thread, no race. qscale carries the attn scale.
        for (unsigned int r = threadIdx.x; r < BR; r += blockDim.x)
        {
            if (q0 + r >= Sq) continue;
            const float qs = qsh[q0 + r];
            int* SrowI = Ssh32 + (size_t)r * BC;
            float* Srow = Ssh + (size_t)r * BC;
            float rowMax = NEG_INF;
            float sDeq[BC];
            for (unsigned int c = 0; c < curBC; c++)
            {
                float s = (float)SrowI[c] * qs * kscale_sh[c];
                sDeq[c] = s;
                if (s > rowMax) rowMax = s;
            }
            const float mPrev = m_state[r];
            const float mNew = fmaxf(mPrev, rowMax);
            const float corr = __expf(mPrev - mNew);
            float rowSum = 0.0f;
            for (unsigned int c = 0; c < curBC; c++)
            {
                float p = __expf(sDeq[c] - mNew);
                Srow[c] = p;
                rowSum += p;
            }
            for (unsigned int c = curBC; c < BC; c++) Srow[c] = 0.0f;
            l_state[r] = l_state[r] * corr + rowSum;
            m_state[r] = mNew;
            float* Orow = Osh + (size_t)r * D;
            for (unsigned int d = 0; d < D; d++) Orow[d] *= corr;
        }
        __syncthreads();

        // O[16 x D] += P[16 x BC] @ V[BC x D], per warp — TF32 path, verbatim flash_attn_v2_tf32.
        for (unsigned int dt = 0; dt < D; dt += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> pf;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> vf;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> of;
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

    for (unsigned int r = threadIdx.x; r < BR; r += blockDim.x)
    {
        if (q0 + r >= Sq) continue;
        const float inv = 1.0f / l_state[r];
        const float* Orow = Osh + (size_t)r * D;
        float* Og = Oh + (size_t)(q0 + r) * D;
        for (unsigned int d = 0; d < D; d++) Og[d] = Orow[d] * inv;
    }
}
