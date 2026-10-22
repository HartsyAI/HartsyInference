// INT8 attention v2: the CPU emulator Int8AttentionV2Reference (HartsyInference.Core) is the executable specification,
// and these kernels reproduce its bytes, scales, means and INT32 products bit for bit. Unmasked, B >= 1, Hkv == Hq,
// head dim 128, key tiles of 128 (BC), strided F16 or F32 operands, F16 or F32 output.
//
// Three launches per call:
//   1. int8v2_stats_*   per (b, h) chunk of 64 keys: the K channel sum in key order and, for per-sequence V scales, the
//                       V channel absmax; the last chunk block of each (b, h) sums the chunk partials in chunk order,
//                       divides by the key count (the emulator's SegmentMean) and folds the V maxima into scales.
//   2. int8v2_quant_*   Q rows (absmax/127 per row), K tiles (smoothed by the mean, one scale per lane class
//                       (key mod 8)/2 per tile) and V tiles (per-channel scales, transposed to [ch][key] with the keys of
//                       every 32-key chunk permuted so a lane's QK accumulators are already its PV A fragment).
//   3. int8v2_attn_*    QK^T on mma.m16n8k32.s8.s8.s32; per key tile the row max is taken in the integer domain and
//                       P = exp2(score - (tileMax - 7.9943534)) is rounded to u8 (the tile max maps to 255); PV on
//                       mma.m16n8k32.u8.s8.s32 into an INT32 partial drained per tile as O = O*oScale + partial*tileScale,
//                       the denominator summed from the same bytes. F16-PV fallback entries run P and V in F16 against
//                       the running max on mma.m16n8k16.f32.f16.f16.f32.
//
// Stats and quant use explicit _rn intrinsics: nvcc contracts a*b+c into an FMA the emulator does not perform.
// NaN propagates through every max (max.NaN.f32), as the emulator's MathF.Max does.
#include <cuda_fp16.h>
#include <stdint.h>

#define V2_D 128
#define V2_BC 128
#define V2_MEAN_CHUNK 64
#define V2_U8_OFFSET 7.9943534f
#define V2_MIN_NORMAL 1.17549435e-38f
#define V2_FULL 0xffffffffu
#define V2_INT_MIN (-2147483647 - 1)
#define V2_NEG_INF __int_as_float(0xff800000)

// Element strides of an operand whose head dim is contiguous; the base pointer already includes the view offset.
struct V2View
{
    unsigned long long batch, head, row;
};

// Where a trace entry writes its (b, h, query tile): QK dots, P bytes, tile max, row sum and INT32 PV partials.
struct V2Trace
{
    int* dots;          // [BR][kvPad]
    unsigned char* p;   // [BR][kvPad], logical key order
    float* tileMax;     // [BR][tiles]
    int* sums;          // [BR][tiles]
    int* partials;      // [BR][tiles][128]
    unsigned b, h, qTile;
};

__device__ __forceinline__ float v2_max_nan(float a, float b)
{
    float r;
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float v2_ex2(float x)
{
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// absmax / range; NaN for a NaN absmax, 1 below the smallest normal float (the emulator's Scale).
__device__ __forceinline__ float v2_scale(float absMax, float range)
{
    if (absMax != absMax) return absMax;
    const float s = __fdiv_rn(absMax, range);
    return s >= V2_MIN_NORMAL ? s : 1.0f;
}

// Round half to even into [-127, 127]; NaN is 0 (the emulator's ToS8).
__device__ __forceinline__ int v2_s8(float x)
{
    if (x != x) return 0;
    return __float2int_rn(fminf(fmaxf(x, -127.0f), 127.0f));
}

__device__ __forceinline__ unsigned v2_pack4(int a, int b, int c, int d)
{
    return (unsigned)(a & 0xff) | ((unsigned)(b & 0xff) << 8) | ((unsigned)(c & 0xff) << 16) | ((unsigned)(d & 0xff) << 24);
}

// An exact int -> float for |x| < 2^22 on the FP32 pipe: the conversion units run at a quarter rate below sm_89.
__device__ __forceinline__ float v2_i2f(int x)
{
    return __fsub_rn(__int_as_float(x + 0x4B400000), 12582912.0f);
}

// The u8 byte of x >= 0 rounded half to even, in the low byte (x + 2^23 lands the integer in the low mantissa bits).
__device__ __forceinline__ unsigned v2_u8_bits(float x)
{
    return __float_as_uint(__fadd_rn(fminf(x, 255.0f), 8388608.0f));
}

__device__ __forceinline__ unsigned v2_pack_low_bytes(unsigned a, unsigned b, unsigned c, unsigned d)
{
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

__device__ __forceinline__ bool v2_finite(float x)
{
    return fabsf(x) <= 3.402823466e38f;
}

template<typename T> __device__ __forceinline__ float v2_ld(const T* p);
template<> __device__ __forceinline__ float v2_ld<__half>(const __half* p) { return __half2float(*p); }
template<> __device__ __forceinline__ float v2_ld<float>(const float* p) { return *p; }

// Four consecutive elements; the dispatcher guarantees 16-byte aligned rows.
template<typename T> __device__ __forceinline__ void v2_ld4(const T* p, float v[4]);
template<> __device__ __forceinline__ void v2_ld4<__half>(const __half* p, float v[4])
{
    const uint2 raw = *(const uint2*)p;
    const float2 lo = __half22float2(*(const __half2*)&raw.x), hi = __half22float2(*(const __half2*)&raw.y);
    v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
}
template<> __device__ __forceinline__ void v2_ld4<float>(const float* p, float v[4])
{
    const float4 raw = *(const float4*)p;
    v[0] = raw.x; v[1] = raw.y; v[2] = raw.z; v[3] = raw.w;
}

// ── Stats launch: grid (chunks, H, B), 256 threads (0-127 K channels, 128-255 V channels) ─────────────────────────────

template<typename T>
__device__ __forceinline__ void v2_stats(const T* __restrict__ K, V2View vk, const T* __restrict__ V, V2View vv,
    float* __restrict__ kPartial, float* __restrict__ vPartial, float* __restrict__ kMean, float* __restrict__ vScale,
    unsigned* __restrict__ counters, unsigned H, unsigned kvLen, unsigned chunks, unsigned seqV)
{
    const unsigned chunk = blockIdx.x, h = blockIdx.y, b = blockIdx.z, bh = b * H + h;
    const unsigned c = threadIdx.x & 127;
    const bool isV = threadIdx.x >= 128;
    const unsigned j0 = chunk * V2_MEAN_CHUNK, j1 = min(j0 + V2_MEAN_CHUNK, kvLen);
    const size_t at = ((size_t)bh * chunks + chunk) * V2_D + c;
    if (!isV)
    {
        const T* p = K + b * vk.batch + h * vk.head + c;
        float partial = 0.0f;
        for (unsigned j = j0; j < j1; j++) partial = __fadd_rn(partial, v2_ld(p + j * vk.row));
        kPartial[at] = partial;
    }
    else if (seqV)
    {
        const T* p = V + b * vv.batch + h * vv.head + c;
        float m = 0.0f;
        for (unsigned j = j0; j < j1; j++) m = v2_max_nan(m, fabsf(v2_ld(p + j * vv.row)));
        vPartial[at] = m;
    }
    // The last chunk block of this (b, h) folds every chunk's partial, in chunk order.
    __shared__ unsigned last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&counters[bh], 1u) == chunks - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (!isV)
    {
        float total = 0.0f;
        for (unsigned i = 0; i < chunks; i++) total = __fadd_rn(total, __ldcg(&kPartial[((size_t)bh * chunks + i) * V2_D + c]));
        kMean[(size_t)bh * V2_D + c] = __fdiv_rn(total, (float)kvLen);
    }
    else if (seqV)
    {
        float m = 0.0f;
        for (unsigned i = 0; i < chunks; i++) m = v2_max_nan(m, __ldcg(&vPartial[((size_t)bh * chunks + i) * V2_D + c]));
        vScale[(size_t)bh * V2_D + c] = v2_scale(m, 127.0f);
    }
    if (threadIdx.x == 0) counters[bh] = 0;
}

extern "C" __global__ void __launch_bounds__(256) int8v2_stats_f16(const __half* __restrict__ K, V2View vk,
    const __half* __restrict__ V, V2View vv, float* kPartial, float* vPartial, float* kMean, float* vScale, unsigned* counters,
    unsigned H, unsigned kvLen, unsigned chunks, unsigned seqV)
{
    v2_stats<__half>(K, vk, V, vv, kPartial, vPartial, kMean, vScale, counters, H, kvLen, chunks, seqV);
}

extern "C" __global__ void __launch_bounds__(256) int8v2_stats_f32(const float* __restrict__ K, V2View vk,
    const float* __restrict__ V, V2View vv, float* kPartial, float* vPartial, float* kMean, float* vScale, unsigned* counters,
    unsigned H, unsigned kvLen, unsigned chunks, unsigned seqV)
{
    v2_stats<float>(K, vk, V, vv, kPartial, vPartial, kMean, vScale, counters, H, kvLen, chunks, seqV);
}

// ── Quant launch: grid (qBlocks + 2·tiles, H, B), 256 threads ─────────────────────────────────────────────────────────
// Blocks [0, qBlocks) quantize 64 Q rows each (a warp per row, a lane per four channels), the next `tiles` blocks one K
// tile each, the last `tiles` blocks one V tile each.

#define V2_Q_ROWS_PER_BLOCK 64

// V layout: [B·H][128 channels][kvPad], keys permuted inside every 32-key chunk: logical r = 16h + 8s + 2t + e lands at
// physical 16h + 4t + 2s + e, the order a lane's m16n8k32 C fragments of four n-tiles form one A fragment.
// pvF16 stores F16 V (scale 1) in natural key order for the F16-PV entries.
template<typename T, bool PV_F16>
__device__ __forceinline__ void v2_quant_v(const T* __restrict__ V, V2View vv, unsigned b, unsigned h, unsigned bh,
    unsigned tile, unsigned kvLen, unsigned tiles, unsigned vTile, const float* __restrict__ vScaleSeq,
    float* __restrict__ vScaleTile, void* __restrict__ v8, unsigned char* smem)
{
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned c = 32 * (warp & 3) + lane, half = warp >> 2;
    const unsigned key0 = tile * V2_BC + 64 * half;
    const size_t kvPad = (size_t)tiles * V2_BC;
    const T* src = V + b * vv.batch + h * vv.head + c;
    float v[64];
    #pragma unroll
    for (unsigned i = 0; i < 64; i++)
        v[i] = key0 + i < kvLen ? v2_ld(src + (key0 + i) * vv.row) : 0.0f;
    if (PV_F16)
    {
        // F16 V in natural key order, [ch][kvPad] halves.
        __half* dst = (__half*)v8 + ((size_t)bh * V2_D + c) * kvPad + key0;
        #pragma unroll
        for (unsigned i = 0; i < 64; i += 2)
            *(__half2*)(dst + i) = __halves2half2(__float2half_rn(v[i]), __float2half_rn(v[i + 1]));
        return;
    }
    float* xmax = (float*)smem;   // [2][128]
    float scale;
    if (vTile)
    {
        float m = 0.0f;
        #pragma unroll
        for (unsigned i = 0; i < 64; i++) m = v2_max_nan(m, fabsf(v[i]));
        xmax[half * V2_D + c] = m;
        __syncthreads();
        scale = v2_scale(v2_max_nan(xmax[c], xmax[V2_D + c]), 127.0f);
        if (half == 0) vScaleTile[((size_t)bh * tiles + tile) * V2_D + c] = scale;
        __syncthreads();
    }
    else
    {
        scale = vScaleSeq[(size_t)bh * V2_D + c];
    }
    const float inv = __frcp_rn(scale);
    // Transpose through shared memory: 128 rows of 128 bytes at a 132-byte stride (conflict-free u32 stores).
    unsigned* rows = (unsigned*)smem;
    #pragma unroll
    for (unsigned grp = 0; grp < 16; grp++)
    {
        // Physical group grp of this half: chunk q = grp / 8, inside it h16 = (grp / 4) % 2, t = grp % 4.
        const unsigned q = grp >> 3, h16 = (grp >> 2) & 1, t = grp & 3;
        const unsigned r = 32 * q + 16 * h16 + 2 * t;
        const int b0 = v2_s8(__fmul_rn(v[r], inv)), b1 = v2_s8(__fmul_rn(v[r + 1], inv));
        const int b2 = v2_s8(__fmul_rn(v[r + 8], inv)), b3 = v2_s8(__fmul_rn(v[r + 9], inv));
        rows[c * 33 + 16 * half + grp] = v2_pack4(b0, b1, b2, b3);
    }
    __syncthreads();
    unsigned char* dst = (unsigned char*)v8 + (size_t)bh * V2_D * kvPad + (size_t)tile * V2_BC;
    for (unsigned row = warp; row < V2_D; row += 8)
        *(unsigned*)(dst + (size_t)row * kvPad + 4 * lane) = rows[row * 33 + lane];
}

template<typename T, bool PV_F16>
__device__ __forceinline__ void v2_quant(const T* __restrict__ Q, V2View vq, const T* __restrict__ K, V2View vk,
    const T* __restrict__ V, V2View vv, signed char* __restrict__ q8, float* __restrict__ qScale, signed char* __restrict__ k8,
    float* __restrict__ kScale, void* __restrict__ v8, float* __restrict__ vScaleTile, const float* __restrict__ kMean,
    const float* __restrict__ vScaleSeq, unsigned H, unsigned Sq, unsigned kvLen, unsigned tiles, unsigned qBlocks,
    unsigned vTile)
{
    __shared__ __align__(16) unsigned char smem[V2_D * 132];
    const unsigned h = blockIdx.y, b = blockIdx.z, bh = b * H + h;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t kvPad = (size_t)tiles * V2_BC;
    if (blockIdx.x < qBlocks)
    {
        const T* base = Q + b * vq.batch + h * vq.head + 4 * lane;
        for (unsigned i = 0; i < V2_Q_ROWS_PER_BLOCK / 8; i++)
        {
            const unsigned row = blockIdx.x * V2_Q_ROWS_PER_BLOCK + warp * (V2_Q_ROWS_PER_BLOCK / 8) + i;
            if (row >= Sq) break;
            float v[4];
            v2_ld4(base + row * vq.row, v);
            float m = v2_max_nan(v2_max_nan(fabsf(v[0]), fabsf(v[1])), v2_max_nan(fabsf(v[2]), fabsf(v[3])));
            #pragma unroll
            for (unsigned off = 16; off > 0; off >>= 1) m = v2_max_nan(m, __shfl_xor_sync(V2_FULL, m, off));
            const float scale = v2_scale(m, 127.0f), inv = __frcp_rn(scale);
            const size_t at = (size_t)bh * Sq + row;
            *(unsigned*)(q8 + at * V2_D + 4 * lane) = v2_pack4(v2_s8(__fmul_rn(v[0], inv)), v2_s8(__fmul_rn(v[1], inv)),
                v2_s8(__fmul_rn(v[2], inv)), v2_s8(__fmul_rn(v[3], inv)));
            if (lane == 0) qScale[at] = scale;
        }
        return;
    }
    const unsigned tileK = blockIdx.x - qBlocks;
    if (tileK >= tiles)
    {
        v2_quant_v<T, PV_F16>(V, vv, b, h, bh, tileK - tiles, kvLen, tiles, vTile, vScaleSeq, vScaleTile, v8, smem);
        return;
    }
    // K tile: warp w holds keys 16w..16w+15, a lane four channels; class of key j is (j & 7) / 2.
    float mean[4];
    {
        const float4 mv = *(const float4*)(kMean + (size_t)bh * V2_D + 4 * lane);
        mean[0] = mv.x; mean[1] = mv.y; mean[2] = mv.z; mean[3] = mv.w;
    }
    const T* base = K + b * vk.batch + h * vk.head + 4 * lane;
    float x[16][4];
    float cls[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll
    for (unsigned i = 0; i < 16; i++)
    {
        const unsigned j = tileK * V2_BC + warp * 16 + i;
        float rowMax = 0.0f;
        if (j < kvLen)
        {
            float v[4];
            v2_ld4(base + j * vk.row, v);
            #pragma unroll
            for (unsigned e = 0; e < 4; e++)
            {
                x[i][e] = __fsub_rn(v[e], mean[e]);
                rowMax = v2_max_nan(rowMax, fabsf(x[i][e]));
            }
        }
        else
        {
            x[i][0] = x[i][1] = x[i][2] = x[i][3] = 0.0f;
        }
        #pragma unroll
        for (unsigned off = 16; off > 0; off >>= 1) rowMax = v2_max_nan(rowMax, __shfl_xor_sync(V2_FULL, rowMax, off));
        cls[(i & 7) >> 1] = v2_max_nan(cls[(i & 7) >> 1], rowMax);
    }
    float* sCls = (float*)smem;   // [8 warps][4]
    if (lane < 4) sCls[warp * 4 + lane] = cls[lane];
    __syncthreads();
    float inv[4];
    #pragma unroll
    for (unsigned k = 0; k < 4; k++)
    {
        float m = 0.0f;
        #pragma unroll
        for (unsigned w = 0; w < 8; w++) m = v2_max_nan(m, sCls[w * 4 + k]);
        const float scale = v2_scale(m, 127.0f);
        inv[k] = __frcp_rn(scale);
        if (threadIdx.x == k) kScale[((size_t)bh * tiles + tileK) * 4 + k] = scale;
    }
    #pragma unroll
    for (unsigned i = 0; i < 16; i++)
    {
        const unsigned j = tileK * V2_BC + warp * 16 + i;
        const float s = inv[(i & 7) >> 1];
        *(unsigned*)(k8 + ((size_t)bh * kvPad + j) * V2_D + 4 * lane) = v2_pack4(v2_s8(__fmul_rn(x[i][0], s)),
            v2_s8(__fmul_rn(x[i][1], s)), v2_s8(__fmul_rn(x[i][2], s)), v2_s8(__fmul_rn(x[i][3], s)));
    }
}

#define V2_QUANT_ENTRY(NAME, T, PV_F16)                                                                                       \
    extern "C" __global__ void __launch_bounds__(256) NAME(const T* __restrict__ Q, V2View vq, const T* __restrict__ K,   \
        V2View vk, const T* __restrict__ V, V2View vv, signed char* q8, float* qScale, signed char* k8, float* kScale,       \
        void* v8, float* vScaleTile, const float* kMean, const float* vScaleSeq, unsigned H, unsigned Sq, unsigned kvLen,    \
        unsigned tiles, unsigned qBlocks, unsigned vTile)                                                                    \
    {                                                                                                                        \
        v2_quant<T, PV_F16>(Q, vq, K, vk, V, vv, q8, qScale, k8, kScale, v8, vScaleTile, kMean, vScaleSeq, H, Sq, kvLen,     \
            tiles, qBlocks, vTile);                                                                                          \
    }

V2_QUANT_ENTRY(int8v2_quant_f16, __half, false)
V2_QUANT_ENTRY(int8v2_quant_f32, float, false)
V2_QUANT_ENTRY(int8v2_quant_f16_pvf16, __half, true)
V2_QUANT_ENTRY(int8v2_quant_f32_pvf16, float, true)

// ── Attention ─────────────────────────────────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ void v2_mma_s8(int (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void v2_mma_u8s8(int (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void v2_mma_f16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void v2_ldsm_x4(unsigned (&r)[4], const void* p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void v2_cp16(void* dst, const void* src)
{
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void v2_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void v2_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

template<typename OutT> __device__ __forceinline__ void v2_store2(OutT* p, float a, float b);
template<> __device__ __forceinline__ void v2_store2<__half>(__half* p, float a, float b)
{
    *(__half2*)p = __halves2half2(__float2half_rn(a), __float2half_rn(b));
}
template<> __device__ __forceinline__ void v2_store2<float>(float* p, float a, float b) { *(float2*)p = make_float2(a, b); }

// Shared memory: K tile [128 keys][128 B] | V tile [128 ch][128 B] (F16 PV: [128 ch][128 keys] halves) | K scales [4] |
// per-tile V scales [128]. Both tiles XOR-swizzle their 16-byte chunks by row & 7, so each 8-row ldmatrix phase covers
// all 32 banks.
template<bool PV_F16>
struct V2Smem
{
    static constexpr unsigned KBytes = V2_BC * V2_D;
    static constexpr unsigned VBytes = V2_D * V2_BC * (PV_F16 ? 2 : 1);
    static constexpr unsigned Total = KBytes + VBytes + 16 + V2_D * 4;
};

// One CTA attends BR query rows of one (b, h) over every key tile; warp w owns rows 16w..16w+15 and a lane rows g, g+8.
// Pipeline per tile t, one K slot and one V slot: wait K(t) + barrier, issue V(t); QK; softmax; wait V(t) + barrier,
// issue K(t+1); PV. V(t) streams in under QK and the softmax, K(t+1) under PV.
template<unsigned BR, bool VTILE, bool PV_F16, typename OutT, bool TRACE>
__device__ __forceinline__ void v2_attn(OutT* __restrict__ out, V2View vo, const signed char* __restrict__ q8,
    const float* __restrict__ qScale, const signed char* __restrict__ k8, const float* __restrict__ kScale,
    const void* __restrict__ v8, const float* __restrict__ vScale, unsigned H, unsigned Sq, unsigned kvLen, unsigned tiles,
    float alphaLog2e, V2Trace trace)
{
    constexpr unsigned THREADS = BR * 2;
    const unsigned qTile = blockIdx.x, h = blockIdx.y, b = blockIdx.z, bh = b * H + h;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const unsigned rowLocal0 = warp * 16 + g, row0 = qTile * BR + rowLocal0, row1 = row0 + 8;
    const size_t kvPad = (size_t)tiles * V2_BC;
    const bool traced = TRACE && b == trace.b && h == trace.h && qTile == trace.qTile;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sK = smem;
    unsigned char* sV = smem + V2Smem<PV_F16>::KBytes;
    float* sKs = (float*)(sV + V2Smem<PV_F16>::VBytes);
    float* sVs = sKs + 4;

    // Q fragments stay in registers: a0 (row g, k 4t4..+3), a1 (row g+8), a2 (row g, +16), a3 (row g+8, +16).
    unsigned qa[4][4];
    {
        const signed char* qh = q8 + (size_t)bh * Sq * V2_D;
        #pragma unroll
        for (unsigned kc = 0; kc < 4; kc++)
        {
            const unsigned d0 = kc * 32 + t4 * 4;
            qa[kc][0] = row0 < Sq ? *(const unsigned*)(qh + (size_t)row0 * V2_D + d0) : 0u;
            qa[kc][1] = row1 < Sq ? *(const unsigned*)(qh + (size_t)row1 * V2_D + d0) : 0u;
            qa[kc][2] = row0 < Sq ? *(const unsigned*)(qh + (size_t)row0 * V2_D + d0 + 16) : 0u;
            qa[kc][3] = row1 < Sq ? *(const unsigned*)(qh + (size_t)row1 * V2_D + d0 + 16) : 0u;
        }
    }
    const float qs0 = row0 < Sq ? __fmul_rn(qScale[(size_t)bh * Sq + row0], alphaLog2e) : 0.0f;
    const float qs1 = row1 < Sq ? __fmul_rn(qScale[(size_t)bh * Sq + row1], alphaLog2e) : 0.0f;

    const signed char* kh = k8 + (size_t)bh * kvPad * V2_D;
    auto loadK = [&](unsigned t)
    {
        const signed char* src = kh + (size_t)t * V2_BC * V2_D;
        #pragma unroll
        for (unsigned i = threadIdx.x; i < V2_BC * 8; i += THREADS)
        {
            const unsigned row = i >> 3, ch = i & 7;
            v2_cp16(sK + row * V2_D + ((ch ^ (row & 7)) << 4), src + (size_t)row * V2_D + ch * 16);
        }
        if (threadIdx.x == 0) v2_cp16(sKs, kScale + ((size_t)bh * tiles + t) * 4);
    };
    auto loadV = [&](unsigned t)
    {
        if (PV_F16)
        {
            // [128 ch][128 keys] halves: 16 chunks of 16 B per row, swizzled within each group of 8 chunks.
            const __half* src = (const __half*)v8 + (size_t)bh * V2_D * kvPad + (size_t)t * V2_BC;
            #pragma unroll
            for (unsigned i = threadIdx.x; i < V2_D * 16; i += THREADS)
            {
                const unsigned row = i >> 4, ch = i & 15;
                v2_cp16(sV + row * 256 + (((ch & 8) | ((ch & 7) ^ (row & 7))) << 4), src + (size_t)row * kvPad + ch * 8);
            }
        }
        else
        {
            const signed char* src = (const signed char*)v8 + (size_t)bh * V2_D * kvPad + (size_t)t * V2_BC;
            #pragma unroll
            for (unsigned i = threadIdx.x; i < V2_D * 8; i += THREADS)
            {
                const unsigned row = i >> 3, ch = i & 7;
                v2_cp16(sV + row * V2_BC + ((ch ^ (row & 7)) << 4), src + (size_t)row * kvPad + ch * 16);
            }
        }
        if (VTILE && threadIdx.x < 32) v2_cp16(sVs + threadIdx.x * 4, vScale + ((size_t)bh * tiles + t) * V2_D + threadIdx.x * 4);
    };

    // ldmatrix row addresses: lane L supplies row L & 7 of matrix L >> 3.
    const unsigned mi = lane >> 3, r8 = lane & 7;
    unsigned kOff[4];
    #pragma unroll
    for (unsigned kc = 0; kc < 4; kc++) kOff[kc] = ((mi >> 1) * 8 + r8) * V2_D + (((2 * kc + (mi & 1)) ^ r8) << 4);
    unsigned vOff[2];
    #pragma unroll
    for (unsigned p = 0; p < 2; p++)
        vOff[p] = PV_F16 ? r8 * 256 + ((((4 * p + mi) & 8) | (((4 * p + mi) & 7) ^ r8)) << 4)
                         : r8 * V2_BC + (((4 * p + mi) ^ r8) << 4);

    float m0 = V2_NEG_INF, m1 = V2_NEG_INF, d0 = 0.0f, d1 = 0.0f;
    bool bad0 = false, bad1 = false;
    float o[16][4];
    #pragma unroll
    for (unsigned dt = 0; dt < 16; dt++) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;

    loadK(0);
    v2_commit();
    for (unsigned t = 0; t < tiles; t++)
    {
        v2_wait_all();
        __syncthreads();
        loadV(t);
        v2_commit();

        int s[16][4];
        #pragma unroll
        for (unsigned nt = 0; nt < 16; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0;
        #pragma unroll
        for (unsigned kc = 0; kc < 4; kc++)
        {
            #pragma unroll
            for (unsigned np = 0; np < 8; np++)
            {
                unsigned kb[4];
                v2_ldsm_x4(kb, sK + np * 16 * V2_D + kOff[kc]);
                v2_mma_s8(s[2 * np], qa[kc], kb[0], kb[1]);
                v2_mma_s8(s[2 * np + 1], qa[kc], kb[2], kb[3]);
            }
        }
        const float ks = sKs[t4];
        const float f0 = __fmul_rn(qs0, ks), f1 = __fmul_rn(qs1, ks);

        // Keys past kvLen in the last tile: out of the max and P = 0.
        const unsigned valid = min(kvLen - t * V2_BC, (unsigned)V2_BC);
        const bool tail = valid < V2_BC;
        if (TRACE && traced)
        {
            #pragma unroll
            for (unsigned nt = 0; nt < 16; nt++)
            {
                const unsigned key = t * V2_BC + nt * 8 + 2 * t4;
                *(int2*)(trace.dots + (size_t)rowLocal0 * kvPad + key) = make_int2(s[nt][0], s[nt][1]);
                *(int2*)(trace.dots + (size_t)(rowLocal0 + 8) * kvPad + key) = make_int2(s[nt][2], s[nt][3]);
            }
        }
        int mx0 = V2_INT_MIN, mx1 = V2_INT_MIN;
        #pragma unroll
        for (unsigned nt = 0; nt < 16; nt++)
        {
            #pragma unroll
            for (unsigned e = 0; e < 2; e++)
            {
                if (tail && nt * 8 + 2 * t4 + e >= valid) { s[nt][e] = V2_INT_MIN; s[nt][2 + e] = V2_INT_MIN; }
                mx0 = max(mx0, s[nt][e]);
                mx1 = max(mx1, s[nt][2 + e]);
            }
        }
        float tileMax0 = mx0 == V2_INT_MIN ? V2_NEG_INF : __fmul_rn(v2_i2f(mx0), f0);
        float tileMax1 = mx1 == V2_INT_MIN ? V2_NEG_INF : __fmul_rn(v2_i2f(mx1), f1);
        #pragma unroll
        for (unsigned off = 1; off <= 2; off <<= 1)
        {
            tileMax0 = v2_max_nan(tileMax0, __shfl_xor_sync(V2_FULL, tileMax0, off));
            tileMax1 = v2_max_nan(tileMax1, __shfl_xor_sync(V2_FULL, tileMax1, off));
        }
        bad0 |= !v2_finite(tileMax0);
        bad1 |= !v2_finite(tileMax1);

        unsigned pa[PV_F16 ? 8 : 4][4];
        float oScale0, oScale1, tileScale0, tileScale1;
        if (!PV_F16)
        {
            const float tm0 = __fsub_rn(tileMax0, V2_U8_OFFSET), tm1 = __fsub_rn(tileMax1, V2_U8_OFFSET);
            unsigned sum0 = 0, sum1 = 0;
            #pragma unroll
            for (unsigned kc = 0; kc < 4; kc++)
            {
                unsigned u[4][4];
                #pragma unroll
                for (unsigned n = 0; n < 4; n++)
                {
                    const unsigned nt = 4 * kc + n;
                    #pragma unroll
                    for (unsigned e = 0; e < 4; e++)
                    {
                        const float sc = __fmul_rn(v2_i2f(s[nt][e]), e < 2 ? f0 : f1);
                        const unsigned bits = v2_u8_bits(v2_ex2(__fsub_rn(sc, e < 2 ? tm0 : tm1)));
                        u[n][e] = s[nt][e] == V2_INT_MIN ? 0u : bits;
                    }
                }
                pa[kc][0] = v2_pack_low_bytes(u[0][0], u[0][1], u[1][0], u[1][1]);
                pa[kc][1] = v2_pack_low_bytes(u[0][2], u[0][3], u[1][2], u[1][3]);
                pa[kc][2] = v2_pack_low_bytes(u[2][0], u[2][1], u[3][0], u[3][1]);
                pa[kc][3] = v2_pack_low_bytes(u[2][2], u[2][3], u[3][2], u[3][3]);
                sum0 = __dp4a(pa[kc][0], 0x01010101u, sum0);
                sum0 = __dp4a(pa[kc][2], 0x01010101u, sum0);
                sum1 = __dp4a(pa[kc][1], 0x01010101u, sum1);
                sum1 = __dp4a(pa[kc][3], 0x01010101u, sum1);
            }
            sum0 += __shfl_xor_sync(V2_FULL, sum0, 1);
            sum0 += __shfl_xor_sync(V2_FULL, sum0, 2);
            sum1 += __shfl_xor_sync(V2_FULL, sum1, 1);
            sum1 += __shfl_xor_sync(V2_FULL, sum1, 2);
            // The emulator's U8Tile: exactly one of oScale and tileScale is below 1.
            const float small0 = v2_ex2(-fabsf(__fsub_rn(m0, tm0))), small1 = v2_ex2(-fabsf(__fsub_rn(m1, tm1)));
            oScale0 = m0 < tm0 ? small0 : 1.0f;
            oScale1 = m1 < tm1 ? small1 : 1.0f;
            tileScale0 = tm0 < m0 ? small0 : 1.0f;
            tileScale1 = tm1 < m1 ? small1 : 1.0f;
            m0 = fmaxf(m0, tm0);
            m1 = fmaxf(m1, tm1);
            d0 = __fmaf_rn(v2_i2f((int)sum0), tileScale0, __fmul_rn(d0, oScale0));
            d1 = __fmaf_rn(v2_i2f((int)sum1), tileScale1, __fmul_rn(d1, oScale1));
            if (TRACE && traced)
            {
                #pragma unroll
                for (unsigned kc = 0; kc < 4; kc++)
                    #pragma unroll
                    for (unsigned n = 0; n < 4; n++)
                    {
                        const unsigned key = t * V2_BC + 32 * kc + 8 * n + 2 * t4;
                        const unsigned w0 = pa[kc][n < 2 ? 0 : 2] >> (16 * (n & 1));
                        const unsigned w1 = pa[kc][n < 2 ? 1 : 3] >> (16 * (n & 1));
                        trace.p[(size_t)rowLocal0 * kvPad + key] = (unsigned char)w0;
                        trace.p[(size_t)rowLocal0 * kvPad + key + 1] = (unsigned char)(w0 >> 8);
                        trace.p[(size_t)(rowLocal0 + 8) * kvPad + key] = (unsigned char)w1;
                        trace.p[(size_t)(rowLocal0 + 8) * kvPad + key + 1] = (unsigned char)(w1 >> 8);
                    }
                if (t4 == 0)
                {
                    trace.tileMax[(size_t)rowLocal0 * tiles + t] = tileMax0;
                    trace.tileMax[(size_t)(rowLocal0 + 8) * tiles + t] = tileMax1;
                    trace.sums[(size_t)rowLocal0 * tiles + t] = (int)sum0;
                    trace.sums[(size_t)(rowLocal0 + 8) * tiles + t] = (int)sum1;
                }
            }
        }
        else
        {
            // F16 P against the running maximum (the emulator's RunningTile with no offset); P·V accumulates into the
            // rescaled O on the tensor cores, F32 throughout.
            const float next0 = fmaxf(m0, tileMax0), next1 = fmaxf(m1, tileMax1);
            oScale0 = v2_ex2(__fsub_rn(m0, next0));
            oScale1 = v2_ex2(__fsub_rn(m1, next1));
            m0 = next0;
            m1 = next1;
            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (unsigned nt = 0; nt < 16; nt++)
            {
                __half p[4];
                #pragma unroll
                for (unsigned e = 0; e < 4; e++)
                {
                    const float sc = __fmul_rn(v2_i2f(s[nt][e]), e < 2 ? f0 : f1);
                    p[e] = s[nt][e] == V2_INT_MIN ? __float2half_rn(0.0f) : __float2half_rn(v2_ex2(__fsub_rn(sc, e < 2 ? m0 : m1)));
                }
                sum0 += __half2float(p[0]) + __half2float(p[1]);
                sum1 += __half2float(p[2]) + __half2float(p[3]);
                const __half2 lo = __halves2half2(p[0], p[1]), hi = __halves2half2(p[2], p[3]);
                // k16 chunk nt/2: a0/a1 hold n-tile 2kc, a2/a3 n-tile 2kc+1.
                pa[nt >> 1][(nt & 1) * 2 + 0] = *(const unsigned*)&lo;
                pa[nt >> 1][(nt & 1) * 2 + 1] = *(const unsigned*)&hi;
            }
            sum0 += __shfl_xor_sync(V2_FULL, sum0, 1);
            sum0 += __shfl_xor_sync(V2_FULL, sum0, 2);
            sum1 += __shfl_xor_sync(V2_FULL, sum1, 1);
            sum1 += __shfl_xor_sync(V2_FULL, sum1, 2);
            d0 = __fadd_rn(__fmul_rn(d0, oScale0), sum0);
            d1 = __fadd_rn(__fmul_rn(d1, oScale1), sum1);
            tileScale0 = tileScale1 = 1.0f;
            #pragma unroll
            for (unsigned dt = 0; dt < 16; dt++)
            {
                o[dt][0] = __fmul_rn(o[dt][0], oScale0); o[dt][1] = __fmul_rn(o[dt][1], oScale0);
                o[dt][2] = __fmul_rn(o[dt][2], oScale1); o[dt][3] = __fmul_rn(o[dt][3], oScale1);
            }
        }

        v2_wait_all();
        __syncthreads();
        if (t + 1 < tiles) loadK(t + 1);
        v2_commit();

        #pragma unroll
        for (unsigned dt = 0; dt < 16; dt++)
        {
            const unsigned c0 = dt * 8 + 2 * t4;
            if (!PV_F16)
            {
                int acc[4] = { 0, 0, 0, 0 };
                #pragma unroll
                for (unsigned p = 0; p < 2; p++)
                {
                    unsigned vb[4];
                    v2_ldsm_x4(vb, sV + dt * 8 * V2_BC + vOff[p]);
                    v2_mma_u8s8(acc, pa[2 * p], vb[0], vb[1]);
                    v2_mma_u8s8(acc, pa[2 * p + 1], vb[2], vb[3]);
                }
                if (TRACE && traced)
                {
                    *(int2*)(trace.partials + ((size_t)rowLocal0 * tiles + t) * V2_D + c0) = make_int2(acc[0], acc[1]);
                    *(int2*)(trace.partials + ((size_t)(rowLocal0 + 8) * tiles + t) * V2_D + c0) = make_int2(acc[2], acc[3]);
                }
                float sc0 = tileScale0, sc1 = tileScale0, sc2 = tileScale1, sc3 = tileScale1;
                if (VTILE)
                {
                    const float2 vs = *(const float2*)(sVs + c0);
                    sc0 = __fmul_rn(tileScale0, vs.x);
                    sc1 = __fmul_rn(tileScale0, vs.y);
                    sc2 = __fmul_rn(tileScale1, vs.x);
                    sc3 = __fmul_rn(tileScale1, vs.y);
                }
                o[dt][0] = __fmaf_rn(v2_i2f(acc[0]), sc0, __fmul_rn(o[dt][0], oScale0));
                o[dt][1] = __fmaf_rn(v2_i2f(acc[1]), sc1, __fmul_rn(o[dt][1], oScale0));
                o[dt][2] = __fmaf_rn(v2_i2f(acc[2]), sc2, __fmul_rn(o[dt][2], oScale1));
                o[dt][3] = __fmaf_rn(v2_i2f(acc[3]), sc3, __fmul_rn(o[dt][3], oScale1));
            }
            else
            {
                #pragma unroll
                for (unsigned p = 0; p < 4; p++)
                {
                    unsigned vb[4];
                    v2_ldsm_x4(vb, sV + dt * 8 * 256 + (p < 2 ? vOff[p] : vOff[p - 2] ^ (8 << 4)));
                    v2_mma_f16(o[dt], pa[2 * p], vb[0], vb[1]);
                    v2_mma_f16(o[dt], pa[2 * p + 1], vb[2], vb[3]);
                }
            }
        }
    }

    const float inv0 = __frcp_rn(d0), inv1 = __frcp_rn(d1);
    OutT* oh = out + b * vo.batch + h * vo.head;
    #pragma unroll
    for (unsigned dt = 0; dt < 16; dt++)
    {
        const unsigned c0 = dt * 8 + 2 * t4;
        float r[4];
        if (VTILE || PV_F16)
        {
            r[0] = __fmul_rn(o[dt][0], inv0); r[1] = __fmul_rn(o[dt][1], inv0);
            r[2] = __fmul_rn(o[dt][2], inv1); r[3] = __fmul_rn(o[dt][3], inv1);
        }
        else
        {
            const float2 vs = *(const float2*)(vScale + (size_t)bh * V2_D + c0);
            r[0] = __fmul_rn(__fmul_rn(o[dt][0], inv0), vs.x); r[1] = __fmul_rn(__fmul_rn(o[dt][1], inv0), vs.y);
            r[2] = __fmul_rn(__fmul_rn(o[dt][2], inv1), vs.x); r[3] = __fmul_rn(__fmul_rn(o[dt][3], inv1), vs.y);
        }
        if (bad0) r[0] = r[1] = __int_as_float(0x7fffffff);
        if (bad1) r[2] = r[3] = __int_as_float(0x7fffffff);
        if (row0 < Sq) v2_store2(oh + row0 * vo.row + c0, r[0], r[1]);
        if (row1 < Sq) v2_store2(oh + row1 * vo.row + c0, r[2], r[3]);
    }
}

#define V2_ATTN_ENTRY(NAME, BR, VTILE, PV_F16, OutT, TRACE, MINB)                                                            \
    extern "C" __global__ void __launch_bounds__(BR * 2, MINB) NAME(OutT* __restrict__ out, V2View vo,                      \
        const signed char* __restrict__ q8, const float* __restrict__ qScale, const signed char* __restrict__ k8,            \
        const float* __restrict__ kScale, const void* __restrict__ v8, const float* __restrict__ vScale, unsigned H,         \
        unsigned Sq, unsigned kvLen, unsigned tiles, float alphaLog2e, V2Trace trace)                                         \
    {                                                                                                                        \
        v2_attn<BR, VTILE, PV_F16, OutT, TRACE>(out, vo, q8, qScale, k8, kScale, v8, vScale, H, Sq, kvLen, tiles, alphaLog2e, \
            trace);                                                                                                          \
    }

V2_ATTN_ENTRY(int8v2_attn_br64_vseq_f16o, 64, false, false, __half, false, 2)
V2_ATTN_ENTRY(int8v2_attn_br64_vtile_f16o, 64, true, false, __half, false, 2)
V2_ATTN_ENTRY(int8v2_attn_br64_vseq_f32o, 64, false, false, float, false, 2)
V2_ATTN_ENTRY(int8v2_attn_br64_vtile_f32o, 64, true, false, float, false, 2)
V2_ATTN_ENTRY(int8v2_attn_br128_vseq_f16o, 128, false, false, __half, false, 1)
V2_ATTN_ENTRY(int8v2_attn_br128_vtile_f16o, 128, true, false, __half, false, 1)
V2_ATTN_ENTRY(int8v2_attn_br128_vseq_f32o, 128, false, false, float, false, 1)
V2_ATTN_ENTRY(int8v2_attn_br128_vtile_f32o, 128, true, false, float, false, 1)
V2_ATTN_ENTRY(int8v2_attn_br64_pvf16_f16o, 64, false, true, __half, false, 2)
V2_ATTN_ENTRY(int8v2_attn_br64_pvf16_f32o, 64, false, true, float, false, 2)
V2_ATTN_ENTRY(int8v2_attn_br128_pvf16_f16o, 128, false, true, __half, false, 1)
V2_ATTN_ENTRY(int8v2_attn_br128_pvf16_f32o, 128, false, true, float, false, 1)
V2_ATTN_ENTRY(int8v2_attn_br64_vseq_trace, 64, false, false, float, true, 2)
V2_ATTN_ENTRY(int8v2_attn_br64_vtile_trace, 64, true, false, float, true, 2)
V2_ATTN_ENTRY(int8v2_attn_br128_vseq_trace, 128, false, false, float, true, 1)
V2_ATTN_ENTRY(int8v2_attn_br128_vtile_trace, 128, true, false, float, true, 1)

// The attention kernels' exp2 for each argument, so a test can replay it through the emulator.
extern "C" __global__ void int8v2_ex2_probe(float* __restrict__ out, const float* __restrict__ in, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v2_ex2(in[i]);
}
