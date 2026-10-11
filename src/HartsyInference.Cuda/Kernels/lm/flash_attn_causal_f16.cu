// FlashAttention-2 forward for LLM prefill: causal (or sliding-window), grouped-query, F16 tensor cores (mma.sync m16n8k16),
// F32 accumulation and online softmax kept in registers. It never materializes the [Tq x Lk] scores.
//
// Q:   [B, Hq, Tq, D]   F32. Converted to F16 when staged.
// K/V: [B, Hkv, Lk, D]  F16. Lk is the buffer's sequence STRIDE; only the first kvLen positions are valid (the host converts the
//      F32 cache to a tight F16 copy first, or passes an F16 cache directly).
// out: [B, Hq, Tq, D]   F32.
// Query head h reads kv head h / kvGroup. Row r of the query block sits at absolute position qOffset + r and attends keys
// 0..qOffset+r (>= qOffset+r-window+1 with a sliding window).
//
// One CTA = 4 warps = 64 query rows (16 per warp); the key axis is walked in tiles of 64. D is a template parameter (64 or 128).
// The CTA index along x is reversed so the long causal rows start first.
//
// Launch: grid = (ceil(Tq / 64), Hq, B), block = 128, dynamic shared memory = 2 * 64 * (D + 8) * 2 bytes.
//
// Build: ../lm/build.sh (nvcc -ptx -arch=sm_80).

#include <cuda_fp16.h>
#include <stdint.h>

#define FA_BR 64
#define FA_BC 64
#define FA_THREADS 128
#define FA_PAD 8
#define FA_NEG_INF (-3.402823466e+38f)

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p)
{
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void mma_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi)
{
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

template <int HD>
__device__ __forceinline__ void fa2_causal(
    float* __restrict__ out, const float* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    int Hq, int Tq, int Hkv, int Lk, int kvLen, int kvGroup, int qOffset, float scale, int window)
{
    constexpr int KS = HD + FA_PAD;   // shared row stride in halves: 8 rows of a ldmatrix tile hit distinct bank groups
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* Ksm = reinterpret_cast<__half*>(smem_raw);
    __half* Vsm = Ksm + FA_BC * KS;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qtile = (int)gridDim.x - 1 - (int)blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    int hkv = h / kvGroup;
    if (hkv >= Hkv) hkv = Hkv - 1;
    const int q0 = qtile * FA_BR;

    // Stage the query tile as F16 (the K region doubles as scratch before the first key tile lands).
    const float* Qh = Q + ((size_t)b * Hq + h) * (size_t)Tq * HD;
    for (int idx = tid; idx < FA_BR * (HD / 4); idx += FA_THREADS) {
        const int row = idx / (HD / 4);
        const int c4 = idx % (HD / 4);
        uint2 packed = make_uint2(0u, 0u);
        if (q0 + row < Tq) {
            const float4 v = *reinterpret_cast<const float4*>(Qh + (size_t)(q0 + row) * HD + c4 * 4);
            packed.x = pack_half2(v.x, v.y);
            packed.y = pack_half2(v.z, v.w);
        }
        *reinterpret_cast<uint2*>(Ksm + row * KS + c4 * 4) = packed;
    }
    __syncthreads();

    uint32_t qf[HD / 16][4];
    #pragma unroll
    for (int ks = 0; ks < HD / 16; ++ks) {
        ldmatrix_x4(qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3],
                    Ksm + (warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + ks * 16 + (lane >> 4) * 8);
    }

    float o[HD / 8][4];
    #pragma unroll
    for (int u = 0; u < HD / 8; ++u) { o[u][0] = o[u][1] = o[u][2] = o[u][3] = 0.0f; }
    float mrow[2] = {FA_NEG_INF, FA_NEG_INF};
    float lrow[2] = {0.0f, 0.0f};

    const int rowLocal0 = warp * 16 + (lane >> 2);
    const int qp0 = qOffset + q0 + rowLocal0;   // absolute position of this thread's first row; the second is +8
    const float sl2 = scale * 1.4426950408889634f;

    const size_t kvBase = ((size_t)b * Hkv + hkv) * (size_t)Lk * HD;
    const int kvEnd = min(kvLen, qOffset + q0 + FA_BR);
    const int kvStart = window > 0 ? max(0, qOffset + q0 - window + 1) : 0;
    const int jStart = kvStart / FA_BC;
    const int jEnd = (kvEnd + FA_BC - 1) / FA_BC;

    for (int j = jStart; j < jEnd; ++j) {
        __syncthreads();   // the previous tile (or the query staging) is no longer read
        const int k0 = j * FA_BC;
        for (int idx = tid; idx < FA_BC * (HD / 8); idx += FA_THREADS) {
            const int row = idx / (HD / 8);
            const int c8 = idx % (HD / 8);
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = make_uint4(0u, 0u, 0u, 0u);
            if (k0 + row < kvLen) {
                const size_t at = kvBase + (size_t)(k0 + row) * HD + c8 * 8;
                kv = *reinterpret_cast<const uint4*>(K + at);
                vv = *reinterpret_cast<const uint4*>(V + at);
            }
            *reinterpret_cast<uint4*>(Ksm + row * KS + c8 * 8) = kv;
            *reinterpret_cast<uint4*>(Vsm + row * KS + c8 * 8) = vv;
        }
        __syncthreads();

        // S = Q K^T for this warp's 16 rows against 64 keys.
        float s[8][4];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.0f;
            #pragma unroll
            for (int ks = 0; ks < HD / 16; ks += 2) {
                uint32_t r0, r1, r2, r3;
                ldmatrix_x4(r0, r1, r2, r3, Ksm + (t * 8 + (lane & 7)) * KS + ks * 16 + (lane >> 3) * 8);
                mma_16816(s[t], qf[ks], r0, r1);
                mma_16816(s[t], qf[ks + 1], r2, r3);
            }
        }

        // Scale to the exp2 domain and mask. Only tiles that touch the causal diagonal, the window edge or the end of the
        // keys need the per-element test.
        const bool needMask = (k0 + FA_BC > qOffset + q0 + rowLocal0 - (lane >> 2)) || (k0 + FA_BC > kvLen)
                              || (window > 0 && k0 < qOffset + q0 + FA_BR - window);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[t][e] * sl2;
                if (needMask) {
                    const int key = k0 + t * 8 + (lane & 3) * 2 + (e & 1);
                    const int qp = qp0 + ((e >> 1) ? 8 : 0);
                    const bool valid = key < kvLen && key <= qp && (window == 0 || key > qp - window);
                    if (!valid) v = FA_NEG_INF;
                }
                s[t][e] = v;
            }
        }

        // Online softmax. A row's 64 scores live in the 4 lanes sharing (lane >> 2).
        float mx0 = FA_NEG_INF, mx1 = FA_NEG_INF;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
            mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mNew0 = fmaxf(mrow[0], mx0), mNew1 = fmaxf(mrow[1], mx1);
        const float use0 = mNew0 == FA_NEG_INF ? 0.0f : mNew0;   // a fully masked row so far keeps exp2 finite
        const float use1 = mNew1 == FA_NEG_INF ? 0.0f : mNew1;
        const float alpha0 = exp2f(mrow[0] - use0), alpha1 = exp2f(mrow[1] - use1);
        mrow[0] = mNew0;
        mrow[1] = mNew1;

        float sum0 = 0.0f, sum1 = 0.0f;
        uint32_t pa[4][4];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const float p0 = exp2f(s[t][0] - use0), p1 = exp2f(s[t][1] - use0);
            const float p2 = exp2f(s[t][2] - use1), p3 = exp2f(s[t][3] - use1);
            sum0 += p0 + p1;
            sum1 += p2 + p3;
            // C-fragment of two adjacent n-tiles is exactly the A-fragment of one 16-key step.
            pa[t >> 1][(t & 1) * 2 + 0] = pack_half2(p0, p1);
            pa[t >> 1][(t & 1) * 2 + 1] = pack_half2(p2, p3);
        }
        lrow[0] = lrow[0] * alpha0 + sum0;
        lrow[1] = lrow[1] * alpha1 + sum1;
        #pragma unroll
        for (int u = 0; u < HD / 8; ++u) {
            o[u][0] *= alpha0; o[u][1] *= alpha0;
            o[u][2] *= alpha1; o[u][3] *= alpha1;
        }

        // O += P V.
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            #pragma unroll
            for (int u = 0; u < HD / 8; u += 2) {
                uint32_t r0, r1, r2, r3;
                ldmatrix_x4_trans(r0, r1, r2, r3, Vsm + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS + u * 8 + (lane >> 4) * 8);
                mma_16816(o[u], pa[kk], r0, r1);
                mma_16816(o[u + 1], pa[kk], r2, r3);
            }
        }
    }

    // Combine the per-lane partial row sums, normalize and write.
    float l0 = lrow[0], l1 = lrow[1];
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const int qi0 = q0 + rowLocal0, qi1 = qi0 + 8;
    float* Oh = out + ((size_t)b * Hq + h) * (size_t)Tq * HD;
    #pragma unroll
    for (int u = 0; u < HD / 8; ++u) {
        const int col = u * 8 + (lane & 3) * 2;
        if (qi0 < Tq) *reinterpret_cast<float2*>(Oh + (size_t)qi0 * HD + col) = make_float2(o[u][0] * inv0, o[u][1] * inv0);
        if (qi1 < Tq) *reinterpret_cast<float2*>(Oh + (size_t)qi1 * HD + col) = make_float2(o[u][2] * inv1, o[u][3] * inv1);
    }
}

extern "C" __global__ void __launch_bounds__(FA_THREADS) lm_fa2_causal_d128(
    float* __restrict__ out, const float* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    int Hq, int Tq, int Hkv, int Lk, int kvLen, int kvGroup, int qOffset, float scale, int window)
{
    fa2_causal<128>(out, Q, K, V, Hq, Tq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, window);
}

extern "C" __global__ void __launch_bounds__(FA_THREADS) lm_fa2_causal_d64(
    float* __restrict__ out, const float* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    int Hq, int Tq, int Hkv, int Lk, int kvLen, int kvGroup, int qOffset, float scale, int window)
{
    fa2_causal<64>(out, Q, K, V, Hq, Tq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, window);
}

// Tight F16 copy of the valid kvLen positions of an F32 key/value cache: dst [rows, kvLen, D], src [rows, Lk, D] (rows = B * Hkv).
// Launch: grid = (rows, ceil(kvLen * D / 4 / 256)), block = 256, one float4 per thread.
extern "C" __global__ void lm_kv_to_f16(
    __half* __restrict__ dstK, __half* __restrict__ dstV,
    const float* __restrict__ srcK, const float* __restrict__ srcV,
    int kvLen, int Lk, int D)
{
    const size_t per = (size_t)kvLen * D / 4;
    const size_t i = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= per) return;
    const size_t row = blockIdx.x;
    const size_t e = i * 4;                                 // element offset inside the row's valid span
    const size_t s = row * (size_t)Lk * D + e;
    const size_t d = row * (size_t)kvLen * D + e;
    const float4 k = *reinterpret_cast<const float4*>(srcK + s);
    const float4 v = *reinterpret_cast<const float4*>(srcV + s);
    *reinterpret_cast<uint2*>(dstK + d) = make_uint2(pack_half2(k.x, k.y), pack_half2(k.z, k.w));
    *reinterpret_cast<uint2*>(dstV + d) = make_uint2(pack_half2(v.x, v.y), pack_half2(v.z, v.w));
}
