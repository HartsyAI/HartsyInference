// INT8 attention v2 with a rotation: the Q and K rows of int8_attn_v2.cu's workspace, rotated before they are quantized as
// the CPU emulator Int8AttentionV2Reference does it (Int8AttentionRotation.Apply), bit for bit. Head dim 128, per-key K
// scales, strided F16 or F32 operands.
//
// One launch, after int8_attn_v2's stats and quant launches and before its attention launch, which reads the same buffers:
//   int8v2_rot_qk_*   Q rows: rotate, absmax/127 per row, bytes. K rows: subtract the stats launch's channel mean, THEN
//                     rotate, absmax/127 per key, bytes. Keys past kvLen get zero bytes and scale 1. V is never rotated,
//                     so the quant launch's V bytes and scales stand.
//
// A warp holds one row, a lane four adjacent channels, so the four-point stage is lane-local and each radix-2 stage of the
// 128-point transform is one lane exchange. Every add, subtract and multiply is an explicit _rn intrinsic: the four-point
// stage's halving feeds the next stage's add, which the compiler would otherwise contract into an FMA.
#include <cuda_fp16.h>

#define ROT_D 128
#define ROT_BC 128
#define ROT_Q_ROWS_PER_BLOCK 64
#define ROT_MIN_NORMAL 1.17549435e-38f
#define ROT_FULL 0xffffffffu
// 128^-1/2 with the four-point stage's halving taken out (Int8AttentionRotation.H128Scale).
#define ROT_H128_SCALE 0.1767766952966369f

// The host's resolved Int8Rotation, as the kernel takes it.
#define ROT_NONE 0u
#define ROT_H4 1u
#define ROT_SIGNED_H128 2u

// Element strides of an operand whose head dim is contiguous; the base pointer already includes the view offset.
struct RotView
{
    unsigned long long batch, head, row;
};

// The larger of two magnitudes, NaN when either is (the emulator's MathF.Max).
__device__ __forceinline__ float rot_max_nan(float a, float b)
{
    if (a != a) return a;
    if (b != b) return b;
    return fmaxf(a, b);
}

// absmax / 127; NaN for a NaN absmax, 1 below the smallest normal float (the emulator's Scale).
__device__ __forceinline__ float rot_scale(float absMax)
{
    if (absMax != absMax) return absMax;
    const float s = __fdiv_rn(absMax, 127.0f);
    return s >= ROT_MIN_NORMAL ? s : 1.0f;
}

// Round half to even into [-127, 127]; NaN is 0 (the emulator's ToS8).
__device__ __forceinline__ int rot_s8(float x)
{
    if (x != x) return 0;
    return __float2int_rn(fminf(fmaxf(x, -127.0f), 127.0f));
}

__device__ __forceinline__ unsigned rot_pack4(int a, int b, int c, int d)
{
    return (unsigned)(a & 0xff) | ((unsigned)(b & 0xff) << 8) | ((unsigned)(c & 0xff) << 16) | ((unsigned)(d & 0xff) << 24);
}

// Four consecutive elements; the dispatcher guarantees 16-byte aligned rows.
template<typename T> __device__ __forceinline__ void rot_ld4(const T* p, float v[4]);
template<> __device__ __forceinline__ void rot_ld4<__half>(const __half* p, float v[4])
{
    const uint2 raw = *(const uint2*)p;
    const float2 lo = __half22float2(*(const __half2*)&raw.x), hi = __half22float2(*(const __half2*)&raw.y);
    v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
}
template<> __device__ __forceinline__ void rot_ld4<float>(const float* p, float v[4])
{
    const float4 raw = *(const float4*)p;
    v[0] = raw.x; v[1] = raw.y; v[2] = raw.z; v[3] = raw.w;
}

// Bit c of word c / 32 set keeps channel c in the signed 128-point transform, clear negates it
// (Int8AttentionRotation.SignWords); a lane's four channels are one nibble.
__device__ __forceinline__ unsigned rot_keep_nibble(unsigned lane)
{
    const unsigned words[4] = { 0x1035997bu, 0x8087f5eeu, 0xee2e4e1au, 0x71132418u };
    return (words[lane >> 3] >> (4 * (lane & 7))) & 15u;
}

// Rotates the row a warp holds in place; x is this lane's channels 4·lane .. 4·lane + 3. Every lane of the warp calls it.
__device__ __forceinline__ void rot_apply(float x[4], unsigned lane, unsigned rotation)
{
    if (rotation == ROT_NONE) return;
    if (rotation == ROT_SIGNED_H128)
    {
        const unsigned keep = rot_keep_nibble(lane);
        #pragma unroll
        for (unsigned e = 0; e < 4; e++)
            if (!((keep >> e) & 1u)) x[e] = -x[e];
    }
    // The four-point stage, halving included: all of H4 and the first stage of the 128-point transform.
    const float a0 = __fadd_rn(x[0], x[1]), a1 = __fsub_rn(x[0], x[1]), a2 = __fadd_rn(x[2], x[3]), a3 = __fsub_rn(x[2], x[3]);
    x[0] = __fmul_rn(__fadd_rn(a0, a2), 0.5f);
    x[1] = __fmul_rn(__fadd_rn(a1, a3), 0.5f);
    x[2] = __fmul_rn(__fsub_rn(a0, a2), 0.5f);
    x[3] = __fmul_rn(__fsub_rn(a1, a3), 0.5f);
    if (rotation == ROT_H4) return;
    // Channel strides 4, 8, .. 64 are lane strides 1, 2, .. 16, ascending: the low index takes lo + hi, the high lo - hi.
    #pragma unroll
    for (unsigned stride = 1; stride < 32; stride <<= 1)
    {
        #pragma unroll
        for (unsigned e = 0; e < 4; e++)
        {
            const float other = __shfl_xor_sync(ROT_FULL, x[e], stride);
            x[e] = (lane & stride) ? __fsub_rn(other, x[e]) : __fadd_rn(x[e], other);
        }
    }
    #pragma unroll
    for (unsigned e = 0; e < 4; e++) x[e] = __fmul_rn(x[e], ROT_H128_SCALE);
}

// The row's scale from its rotated channels, and this lane's four bytes packed with the first channel in the low byte.
__device__ __forceinline__ unsigned rot_quantize(const float x[4], float& scale)
{
    float m = rot_max_nan(rot_max_nan(fabsf(x[0]), fabsf(x[1])), rot_max_nan(fabsf(x[2]), fabsf(x[3])));
    #pragma unroll
    for (unsigned off = 16; off > 0; off >>= 1) m = rot_max_nan(m, __shfl_xor_sync(ROT_FULL, m, off));
    scale = rot_scale(m);
    const float inv = __frcp_rn(scale);
    return rot_pack4(rot_s8(__fmul_rn(x[0], inv)), rot_s8(__fmul_rn(x[1], inv)), rot_s8(__fmul_rn(x[2], inv)),
        rot_s8(__fmul_rn(x[3], inv)));
}

// Grid (qBlocks + tiles, H, B), 256 threads. Blocks [0, qBlocks) take 64 Q rows each (eight per warp), the next `tiles`
// blocks one K tile each (warp w holds keys 16w .. 16w + 15).
template<typename T>
__device__ __forceinline__ void rot_qk(const T* __restrict__ Q, RotView vq, const T* __restrict__ K, RotView vk,
    signed char* __restrict__ q8, float* __restrict__ qScale, signed char* __restrict__ k8, float* __restrict__ kScale,
    const float* __restrict__ kMean, unsigned H, unsigned Sq, unsigned kvLen, unsigned tiles, unsigned qBlocks,
    unsigned rotation)
{
    const unsigned h = blockIdx.y, b = blockIdx.z, bh = b * H + h;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (blockIdx.x < qBlocks)
    {
        const T* base = Q + b * vq.batch + h * vq.head + 4 * lane;
        constexpr unsigned ROWS = ROT_Q_ROWS_PER_BLOCK / 8;
        const unsigned row0 = blockIdx.x * ROT_Q_ROWS_PER_BLOCK + warp * ROWS;
        #pragma unroll 1
        for (unsigned i = 0; i < ROWS; i++)
        {
            const unsigned row = row0 + i;
            if (row >= Sq) break;
            float x[4];
            rot_ld4(base + row * vq.row, x);
            rot_apply(x, lane, rotation);
            float scale;
            const unsigned bytes = rot_quantize(x, scale);
            const size_t at = (size_t)bh * Sq + row;
            *(unsigned*)(q8 + at * ROT_D + 4 * lane) = bytes;
            if (lane == 0) qScale[at] = scale;
        }
        return;
    }
    const unsigned tile = blockIdx.x - qBlocks;
    const size_t kvPad = (size_t)tiles * ROT_BC;
    float mean[4];
    {
        const float4 mv = *(const float4*)(kMean + (size_t)bh * ROT_D + 4 * lane);
        mean[0] = mv.x; mean[1] = mv.y; mean[2] = mv.z; mean[3] = mv.w;
    }
    const T* base = K + b * vk.batch + h * vk.head + 4 * lane;
    #pragma unroll 1
    for (unsigned i = 0; i < 16; i++)
    {
        const unsigned j = tile * ROT_BC + warp * 16 + i;
        float x[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (j < kvLen)
        {
            float v[4];
            rot_ld4(base + j * vk.row, v);
            #pragma unroll
            for (unsigned e = 0; e < 4; e++) x[e] = __fsub_rn(v[e], mean[e]);
            rot_apply(x, lane, rotation);
        }
        float scale;
        const unsigned bytes = rot_quantize(x, scale);
        *(unsigned*)(k8 + ((size_t)bh * kvPad + j) * ROT_D + 4 * lane) = bytes;
        if (lane == 0) kScale[(size_t)bh * kvPad + j] = scale;
    }
}

#define ROT_QK_ENTRY(NAME, T)                                                                                                \
    extern "C" __global__ void __launch_bounds__(256) NAME(const T* __restrict__ Q, RotView vq, const T* __restrict__ K,     \
        RotView vk, signed char* q8, float* qScale, signed char* k8, float* kScale, const float* kMean, unsigned H,          \
        unsigned Sq, unsigned kvLen, unsigned tiles, unsigned qBlocks, unsigned rotation)                                    \
    {                                                                                                                        \
        rot_qk<T>(Q, vq, K, vk, q8, qScale, k8, kScale, kMean, H, Sq, kvLen, tiles, qBlocks, rotation);                      \
    }

ROT_QK_ENTRY(int8v2_rot_qk_f16, __half)
ROT_QK_ENTRY(int8v2_rot_qk_f32, float)
