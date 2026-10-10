// Flash-decoding partials for ONE query row per sequence (LLM decode), grouped-query aware: a block reads each key/value row of its
// KV head ONCE and serves every query head that shares it, so the cache is streamed kvGroup times less than one block per query head.
//
// Same partial-state contract as flash_attn_f32_split.cu (so its combine kernel merges the chunks), idx = b * Hq + h, split g in [0, G):
//   partialM[idx*G + g]            running max of the (scaled, soft-capped) scores of chunk g
//   partialL[idx*G + g]            sum of exp(score - m)
//   partialAcc[(idx*G + g)*D + d]  sum of exp(score - m) * V, un-normalized
// An empty chunk writes m = -INF, l = 0, acc = 0, which the combine weighs as zero.
//
// Causal for the single row at absolute position qOffset (keys 0..qOffset), optionally windowed to the last `window` keys and soft-capped.
// dPos, when non-null, supplies {kvLen, qOffset} from device memory so one captured CUDA graph replays at every position.
//
// Block = 256 threads = 8 warps; warp w owns the heads w, w+8, ... of the group. The key axis is walked in tiles of 32: a tile of K and V
// is staged in shared memory (coalesced), then each warp takes one key per lane for the scores and one head dimension slice per lane for
// the weighted sum. D is a template parameter (64 or 128); the group can be up to 16 heads.
//
// Launch: grid = (G, Hkv, B), block = 256, dynamic shared memory = DecodeSharedBytes(D) (see CudaKernels.Fa2.cs).
//
// Build: ../lm/build.sh (nvcc -ptx -arch=sm_80).

#include <cuda_fp16.h>
#include <stdint.h>

#define DEC_BK 32
#define DEC_WARPS 8
#define DEC_THREADS 256
#define DEC_MAX_GROUP 16
#define DEC_INF 3.402823466e+38f

__device__ __forceinline__ void dec_load4(const float* p, float4& o) { o = *reinterpret_cast<const float4*>(p); }

__device__ __forceinline__ void dec_load4(const __half* p, float4& o)
{
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const __half2 a = *reinterpret_cast<const __half2*>(&raw.x);
    const __half2 b = *reinterpret_cast<const __half2*>(&raw.y);
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    o = make_float4(fa.x, fa.y, fb.x, fb.y);
}

template <typename KvT, int D>
__device__ __forceinline__ void fa_decode_gqa(
    float* __restrict__ partialM, float* __restrict__ partialL, float* __restrict__ partialAcc,
    const float* __restrict__ Q, const KvT* __restrict__ K, const KvT* __restrict__ V,
    int Hq, int Hkv, int Lk, int kvLen, int kvGroup, int qOffset, float scale, float softcap, int window,
    int G, int chunk, const int* __restrict__ dPos)
{
    constexpr int KS = D + 4;   // K row stride in floats: float4 reads of 8 consecutive rows hit distinct bank groups
    constexpr int DL = D / 32;  // output dimensions per lane
    extern __shared__ __align__(16) float smem[];
    float* Qs = smem;                          // DEC_MAX_GROUP * D
    float* Ks = Qs + DEC_MAX_GROUP * D;        // DEC_BK * KS
    float* Vs = Ks + DEC_BK * KS;              // DEC_BK * D
    float* Ps = Vs + DEC_BK * D;               // DEC_WARPS * DEC_BK

    const int g = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (dPos != nullptr) { kvLen = dPos[0]; qOffset = dPos[1]; }

    const int kMax = min(qOffset, kvLen - 1);
    const int kMin = window > 0 ? max(0, qOffset - window + 1) : 0;
    const int cStart = g * chunk;
    const int cEnd = min(cStart + chunk, kvLen) - 1;
    const int kStart = max(cStart, kMin);
    const int kEnd = min(cEnd, kMax);

    const int group = min(kvGroup, DEC_MAX_GROUP);
    const int firstHead = hk * kvGroup;
    for (int i = tid; i < group * D; i += DEC_THREADS) {
        const int hi = i / D, d = i % D;
        Qs[hi * D + d] = Q[((size_t)b * Hq + firstHead + hi) * D + d];
    }

    constexpr int SLOTS = (DEC_MAX_GROUP + DEC_WARPS - 1) / DEC_WARPS;   // head slots per warp
    float m[SLOTS], l[SLOTS], acc[SLOTS][DL];
    #pragma unroll
    for (int j = 0; j < SLOTS; ++j) {
        m[j] = -DEC_INF; l[j] = 0.0f;
        #pragma unroll
        for (int e = 0; e < DL; ++e) acc[j][e] = 0.0f;
    }

    const size_t kvBase = ((size_t)b * Hkv + hk) * (size_t)Lk * D;
    for (int k0 = kStart; k0 <= kEnd; k0 += DEC_BK) {
        __syncthreads();   // the previous tile (and the query staging) are no longer read
        for (int i = tid; i < DEC_BK * (D / 4); i += DEC_THREADS) {
            const int row = i / (D / 4), c4 = i % (D / 4);
            float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f), vv4 = kv4;
            if (k0 + row <= kEnd) {
                const size_t at = kvBase + (size_t)(k0 + row) * D + c4 * 4;
                dec_load4(K + at, kv4);
                dec_load4(V + at, vv4);
            }
            *reinterpret_cast<float4*>(Ks + row * KS + c4 * 4) = kv4;
            *reinterpret_cast<float4*>(Vs + row * D + c4 * 4) = vv4;
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < SLOTS; ++j) {
            const int hi = warp + j * DEC_WARPS;
            if (hi >= group) continue;
            const float* qh = Qs + hi * D;
            const float* kr = Ks + lane * KS;
            float s = 0.0f;
            #pragma unroll
            for (int d4 = 0; d4 < D / 4; ++d4) {
                const float4 kk = *reinterpret_cast<const float4*>(kr + d4 * 4);
                const float4 qq = *reinterpret_cast<const float4*>(qh + d4 * 4);
                s += qq.x * kk.x + qq.y * kk.y + qq.z * kk.z + qq.w * kk.w;
            }
            s *= scale;
            if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
            const bool valid = k0 + lane <= kEnd;
            if (!valid) s = -DEC_INF;

            float tm = s;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, o));
            const float mNew = fmaxf(m[j], tm);
            const float corr = m[j] <= -DEC_INF ? 0.0f : __expf(m[j] - mNew);
            const float p = valid ? __expf(s - mNew) : 0.0f;
            float ps = p;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
            l[j] = l[j] * corr + ps;
            m[j] = mNew;

            Ps[warp * DEC_BK + lane] = p;
            __syncwarp();
            #pragma unroll
            for (int e = 0; e < DL; ++e) acc[j][e] *= corr;
            #pragma unroll 8
            for (int kk = 0; kk < DEC_BK; ++kk) {
                const float pk = Ps[warp * DEC_BK + kk];
                if constexpr (DL == 4) {
                    const float4 vv = *reinterpret_cast<const float4*>(Vs + kk * D + lane * 4);
                    acc[j][0] += pk * vv.x; acc[j][1] += pk * vv.y; acc[j][2] += pk * vv.z; acc[j][3] += pk * vv.w;
                } else {
                    const float2 vv = *reinterpret_cast<const float2*>(Vs + kk * D + lane * 2);
                    acc[j][0] += pk * vv.x; acc[j][1] += pk * vv.y;
                }
            }
            __syncwarp();
        }
    }

    #pragma unroll
    for (int j = 0; j < SLOTS; ++j) {
        const int hi = warp + j * DEC_WARPS;
        if (hi >= group) continue;
        const size_t idx = (size_t)b * Hq + firstHead + hi;
        const size_t pBase = idx * G + g;
        if (lane == 0) { partialM[pBase] = m[j]; partialL[pBase] = l[j]; }
        float* dst = partialAcc + pBase * D + lane * DL;
        if constexpr (DL == 4) *reinterpret_cast<float4*>(dst) = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        else *reinterpret_cast<float2*>(dst) = make_float2(acc[j][0], acc[j][1]);
    }
}

#define DEC_ARGS float* __restrict__ partialM, float* __restrict__ partialL, float* __restrict__ partialAcc, \
    const float* __restrict__ Q, int Hq, int Hkv, int Lk, int kvLen, int kvGroup, int qOffset, float scale, float softcap, \
    int window, int G, int chunk, const int* __restrict__ dPos

extern "C" __global__ void __launch_bounds__(DEC_THREADS) lm_fa_decode_gqa_f32_d128(DEC_ARGS, const float* __restrict__ K, const float* __restrict__ V)
{ fa_decode_gqa<float, 128>(partialM, partialL, partialAcc, Q, K, V, Hq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, softcap, window, G, chunk, dPos); }

extern "C" __global__ void __launch_bounds__(DEC_THREADS) lm_fa_decode_gqa_f32_d64(DEC_ARGS, const float* __restrict__ K, const float* __restrict__ V)
{ fa_decode_gqa<float, 64>(partialM, partialL, partialAcc, Q, K, V, Hq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, softcap, window, G, chunk, dPos); }

extern "C" __global__ void __launch_bounds__(DEC_THREADS) lm_fa_decode_gqa_f16_d128(DEC_ARGS, const __half* __restrict__ K, const __half* __restrict__ V)
{ fa_decode_gqa<__half, 128>(partialM, partialL, partialAcc, Q, K, V, Hq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, softcap, window, G, chunk, dPos); }

extern "C" __global__ void __launch_bounds__(DEC_THREADS) lm_fa_decode_gqa_f16_d64(DEC_ARGS, const __half* __restrict__ K, const __half* __restrict__ V)
{ fa_decode_gqa<__half, 64>(partialM, partialL, partialAcc, Q, K, V, Hq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, softcap, window, G, chunk, dPos); }
