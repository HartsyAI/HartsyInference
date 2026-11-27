// Flash-decoding partials for ONE query row of ONE sequence whose keys and values live in a page pool: flash_attn_decode_gqa.cu
// with the key/value row found through a page table. The pool buffers are [1, Hkv, poolRows, D] (F32), page p is rows
// [p * pageSize, (p + 1) * pageSize) of every head, and row r of the sequence is pool row
// (pageTable[r >> pageShift] << pageShift) + (r & (pageSize - 1)). pageSize is a power of two.
//
// Everything else is that kernel's: the same partial-state contract (so its combine kernel merges the chunks), the same tiles of
// 32 keys, the same order of arithmetic. For the same rows it writes the same partials bit for bit, which is what lets a paged
// sequence decode to the same tokens as a contiguous one. The body is a copy and not a shared template because the contiguous
// kernel ships as nvcc output whose bytes its timings were taken on; fold the two when a box with that nvcc can show the
// contiguous PTX unchanged.
//
// No device positions: the page table changes as the sequence grows, so this kernel is not replayed from a captured graph.
//
// Launch: grid = (G, Hkv, 1), block = 256, dynamic shared memory = DecodeSharedBytes(D) (see CudaKernels.Fa2.cs).
//
// Build: ../lm/build.sh (nvrtc).

#include <cuda_fp16.h>
#if defined(__CUDACC_RTC__)
typedef unsigned int uint32_t;   // NVRTC has no host <stdint.h>; the kernels only use uint32_t
#else
#include <stdint.h>
#endif

#define DEC_BK 32
#define DEC_WARPS 8
#define DEC_THREADS 256
#define DEC_MAX_GROUP 16
#define DEC_INF 3.402823466e+38f

// The raw vector one thread moves per step: four elements of the cache's own type, kept as stored in shared memory.
template <typename T> struct DecRaw;
template <> struct DecRaw<float> { using type = float4; };
template <> struct DecRaw<__half> { using type = uint2; };

__device__ __forceinline__ float4 dec_f4(const float4& r) { return r; }
__device__ __forceinline__ float4 dec_f4(const uint2& r)
{
    const __half2 a = *reinterpret_cast<const __half2*>(&r.x);
    const __half2 b = *reinterpret_cast<const __half2*>(&r.y);
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    return make_float4(fa.x, fa.y, fb.x, fb.y);
}
__device__ __forceinline__ float2 dec_ld2(const float* p) { return *reinterpret_cast<const float2*>(p); }
__device__ __forceinline__ float2 dec_ld2(const __half* p) { return __half22float2(*reinterpret_cast<const __half2*>(p)); }

template <typename KvT, int D>
__device__ __forceinline__ void fa_decode_gqa(
    float* __restrict__ partialM, float* __restrict__ partialL, float* __restrict__ partialAcc,
    const float* __restrict__ Q, const KvT* __restrict__ K, const KvT* __restrict__ V,
    int Hq, int Hkv, int Lk, int kvLen, int kvGroup, int qOffset, float scale, float softcap, int window,
    int G, int chunk, const int* __restrict__ pageTable, int pageShift)
{
    constexpr int KS = D + 4;   // K row stride in elements: 8-16 byte reads of consecutive rows hit distinct banks
    constexpr int DL = D / 32;  // output dimensions per lane
    using Raw = typename DecRaw<KvT>::type;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int group = min(kvGroup, DEC_MAX_GROUP);
    float* Qs = reinterpret_cast<float*>(smem_raw);              // group * D floats
    float* Ps = Qs + group * D;                                   // DEC_WARPS * DEC_BK floats
    KvT* Ks = reinterpret_cast<KvT*>(Ps + DEC_WARPS * DEC_BK);    // DEC_BK * KS, the cache's element type
    KvT* Vs = Ks + DEC_BK * KS;                                   // DEC_BK * D

    const int g = blockIdx.x;
    const int hk = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int pageMask = (1 << pageShift) - 1;

    const int kMax = min(qOffset, kvLen - 1);
    const int kMin = window > 0 ? max(0, qOffset - window + 1) : 0;
    const int cStart = g * chunk;
    const int cEnd = min(cStart + chunk, kvLen) - 1;
    const int kStart = max(cStart, kMin);
    const int kEnd = min(cEnd, kMax);

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

    // The next tile is fetched into registers while the current one is computed, so the key/value stream is never idle behind the math.
    constexpr int LOADS = DEC_BK * (D / 4) / DEC_THREADS;
    Raw rk[LOADS], rv[LOADS];
    auto fetch = [&](int k0) {
        #pragma unroll
        for (int q = 0; q < LOADS; ++q) {
            const int i = tid + q * DEC_THREADS;
            const int row = i / (D / 4), c4 = i % (D / 4);
            rk[q] = Raw{};
            rv[q] = Raw{};
            if (k0 + row <= kEnd) {
                const int r = k0 + row;
                const size_t poolRow = ((size_t)pageTable[r >> pageShift] << pageShift) + (size_t)(r & pageMask);
                const size_t at = kvBase + poolRow * D + c4 * 4;
                rk[q] = *reinterpret_cast<const Raw*>(K + at);
                rv[q] = *reinterpret_cast<const Raw*>(V + at);
            }
        }
    };
    if (kStart <= kEnd) fetch(kStart);
    for (int k0 = kStart; k0 <= kEnd; k0 += DEC_BK) {
        __syncthreads();   // the previous tile (and the query staging) are no longer read
        #pragma unroll
        for (int q = 0; q < LOADS; ++q) {
            const int i = tid + q * DEC_THREADS;
            const int row = i / (D / 4), c4 = i % (D / 4);
            *reinterpret_cast<Raw*>(Ks + row * KS + c4 * 4) = rk[q];
            *reinterpret_cast<Raw*>(Vs + row * D + c4 * 4) = rv[q];
        }
        if (k0 + DEC_BK <= kEnd) fetch(k0 + DEC_BK);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < SLOTS; ++j) {
            const int hi = warp + j * DEC_WARPS;
            if (hi >= group) continue;
            const float* qh = Qs + hi * D;
            const KvT* kr = Ks + lane * KS;
            float s = 0.0f;
            #pragma unroll
            for (int d4 = 0; d4 < D / 4; ++d4) {
                const float4 kk = dec_f4(*reinterpret_cast<const Raw*>(kr + d4 * 4));
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
            const float corr = l[j] == 0.0f ? 0.0f : __expf(m[j] - mNew);
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
                    const float4 vv = dec_f4(*reinterpret_cast<const Raw*>(Vs + kk * D + lane * 4));
                    acc[j][0] += pk * vv.x; acc[j][1] += pk * vv.y; acc[j][2] += pk * vv.z; acc[j][3] += pk * vv.w;
                } else {
                    const float2 vv = dec_ld2(Vs + kk * D + lane * 2);
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

#define DEC_PAGED_ARGS float* __restrict__ partialM, float* __restrict__ partialL, float* __restrict__ partialAcc, \
    const float* __restrict__ Q, int Hq, int Hkv, int Lk, int kvLen, int kvGroup, int qOffset, float scale, float softcap, \
    int window, int G, int chunk, const int* __restrict__ pageTable, int pageShift, \
    const float* __restrict__ K, const float* __restrict__ V

extern "C" __global__ void __launch_bounds__(DEC_THREADS) lm_fa_decode_gqa_paged_f32_d128(DEC_PAGED_ARGS)
{ fa_decode_gqa<float, 128>(partialM, partialL, partialAcc, Q, K, V, Hq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, softcap, window, G, chunk, pageTable, pageShift); }

extern "C" __global__ void __launch_bounds__(DEC_THREADS) lm_fa_decode_gqa_paged_f32_d64(DEC_PAGED_ARGS)
{ fa_decode_gqa<float, 64>(partialM, partialL, partialAcc, Q, K, V, Hq, Hkv, Lk, kvLen, kvGroup, qOffset, scale, softcap, window, G, chunk, pageTable, pageShift); }
