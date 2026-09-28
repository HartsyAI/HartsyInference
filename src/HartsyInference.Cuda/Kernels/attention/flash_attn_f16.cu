// FlashAttention-2-style fused attention for F16 Q/K/V/O with F32 accumulation, register-resident (raw mma.sync,
// architecturally defined fragment layouts — the same discipline as sage_attn_int8_v1.cu, with an f16 QK^T in
// place of the s8 one). Exists for head dims the cuDNN fused engine serves slowly or not at all: at D=256 on Ada
// cuDNN offers one engine config at ~80 TFLOPS, where this layout reaches the FA2 regime.
//
// Every tensor is addressed through explicit (batch, head, row) element strides, so one kernel reads head-major
// [B,H,S,D], token-major [S, H*D], or rows inside a wider fused projection without a copy. Rows must start on a
// 16-byte boundary (strides and base a multiple of 8 halves) — the dispatcher checks.
//
// Tiling: BR=128 query rows per block (8 warps × 16), BC=32 keys per step, 256 threads. Two warps per scheduler
// is what hides the mma and ldmatrix latency: a 4-warp block (one warp per scheduler) measured 113 TFLOPS at D=256,
// stalled mostly on the tensor pipe with nothing else eligible. The Q tile is staged in shared memory once and re-read per k-chunk through ldmatrix (at
// D=256 a register-resident Q would cost 64 regs on top of the 128-reg O accumulator). K and V each have ONE
// buffer, pipelined FA2-style: V(j) streams in while S(j) and its softmax compute, K(j+1) while P·V(j) computes.
// Tails are zero-filled by cp.async. Shared tiles are XOR-swizzled at 16-byte granularity so each 8-row ldmatrix
// phase touches all 32 banks. Shared memory = (BR + 2·BC)·D halves: 96 KB at D=256 (needs the max-dynamic-shared
// opt-in), 48 KB at D=128. 256 threads × ≤ 255 registers fits the 64 K register file, so one block per SM.
//
// Per k-step per warp: S = Q·Kᵀ as (BC/8) n-tiles × (D/16) k-chunks of mma.m16n8k16 (f32 acc); online softmax in
// the log2 domain (scale·log2e folded by the caller) on the C fragments; P repacked to f16 A fragments with no
// shuffle; O += P·V as (D/8) n-tiles × (BC/16) k-chunks. Key columns ≥ Skv are −inf before the row max; query
// rows ≥ Sq compute on zeros and are never stored.
//
// Build: nvcc -ptx -arch=sm_80 flash_attn_f16.cu (PTX must say .version 9.0 — pinned toolchain).
#include <cuda_fp16.h>

#define FA_WARPS 8
#define FA_BR (16 * FA_WARPS)
#define FA_BC 32
#define FA_THREADS (32 * FA_WARPS)
#define FA_NEG_INF (-3.402823466e+38f)

struct FaStrides
{
    unsigned long long batch, head, row;   // element strides
};

__device__ __forceinline__ void fa_mma(float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void fa_ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const __half* p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}

__device__ __forceinline__ void fa_ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const __half* p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}

// 16-byte async copy; valid=false writes zeros (src-size 0 reads nothing, so src only has to be a legal address).
__device__ __forceinline__ void fa_cp16(void* dst, const void* src, bool valid)
{
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void fa_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void fa_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// Swizzled offset (in halves) of element (row, col) in a [rows][D] tile: 16-byte chunks XORed with row & 7.
template<unsigned D>
__device__ __forceinline__ unsigned fa_off(unsigned row, unsigned col)
{
    return row * D + ((((col >> 3) ^ (row & 7u))) << 3) + (col & 7u);
}

// Copies ROWS rows of D halves starting at global row r0 into a swizzled shared tile; rows ≥ limit are zeroed.
// A thread always copies the same 16-byte column chunk, and rows advance by a multiple of 8 per pass, so its
// swizzled shared column is fixed and the whole walk is one pointer increment per pass.
template<unsigned D, unsigned ROWS>
__device__ __forceinline__ void fa_load_tile(__half* tile, const __half* base, unsigned long long rowStride,
    unsigned r0, unsigned limit)
{
    constexpr unsigned chunks = D / 8, pass = FA_THREADS / chunks;
    static_assert(pass % 8 == 0 && ROWS % pass == 0, "tile walk assumes whole passes of a multiple of 8 rows");
    const unsigned r = threadIdx.x / chunks, c = threadIdx.x % chunks;
    __half* dst = tile + fa_off<D>(r, c * 8);
    const __half* src = base + (unsigned long long)(r0 + r) * rowStride + c * 8;
    #pragma unroll
    for (unsigned it = 0; it < ROWS / pass; it++)
    {
        const bool valid = r0 + r + it * pass < limit;
        fa_cp16(dst + it * pass * D, valid ? src : base, valid);
        src += (unsigned long long)pass * rowStride;
    }
}

template<unsigned D>
__device__ __forceinline__ void flash_attn_f16_core(
    __half* __restrict__ O, const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
    FaStrides so, FaStrides sq, FaStrides sk, FaStrides sv,
    unsigned int Sq, unsigned int Skv, float scaleLog2)
{
    const unsigned q0 = blockIdx.x * FA_BR;
    if (q0 >= Sq) return;
    const unsigned h = blockIdx.y, b = blockIdx.z;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned g = lane >> 2, t4 = lane & 3;
    constexpr unsigned dTiles = D / 8, kChunks = D / 16;
    constexpr unsigned nTiles = FA_BC / 8, pChunks = FA_BC / 16;

    const __half* Qh = Q + b * sq.batch + h * sq.head;
    const __half* Kh = K + b * sk.batch + h * sk.head;
    const __half* Vh = V + b * sv.batch + h * sv.head;
    __half* Oh = O + b * so.batch + h * so.head;

    extern __shared__ __align__(16) unsigned char faSmem[];
    __half* Qs = (__half*)faSmem;
    __half* Kst = Qs + FA_BR * D;
    __half* Vst = Kst + FA_BC * D;

    const unsigned nSteps = (Skv + FA_BC - 1) / FA_BC;
    fa_load_tile<D, FA_BR>(Qs, Qh, sq.row, q0, Sq);
    fa_load_tile<D, FA_BC>(Kst, Kh, sk.row, 0, Skv);
    fa_commit();

    float o[dTiles][4];
    #pragma unroll
    for (unsigned dt = 0; dt < dTiles; dt++) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
    float m0 = FA_NEG_INF, m1 = FA_NEG_INF, l0 = 0.0f, l1 = 0.0f;

    // ldmatrix lane addressing (see the header): A = Q rows (lane & 15), col half (lane >> 4); B for QKᵀ = K rows
    // (lane & 7) + 8·(lane >> 4), col half ((lane >> 3) & 1); B for PV (.trans) = V rows (lane & 7) + 8·((lane >> 3) & 1),
    // col half (lane >> 4).
    const unsigned qRow = warp * 16 + (lane & 15), qCol = (lane >> 4) * 8;
    const unsigned kRow = (lane & 7) + ((lane >> 4) << 3), kCol = ((lane >> 3) & 1) * 8;
    const unsigned vRow = (lane & 7) + (((lane >> 3) & 1) << 3), vCol = (lane >> 4) * 8;

    for (unsigned step = 0; step < nSteps; step++)
    {
        fa_wait<0>();
        __syncthreads();   // K(step) (and Q) landed; every warp is past P·V(step−1), so V's buffer is free
        fa_load_tile<D, FA_BC>(Vst, Vh, sv.row, step * FA_BC, Skv);
        fa_commit();
        const unsigned cur = min((unsigned)FA_BC, Skv - step * FA_BC);

        // ── S = Q·Kᵀ ──
        float s[nTiles][4];
        #pragma unroll
        for (unsigned nt = 0; nt < nTiles; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
        #pragma unroll
        for (unsigned kc = 0; kc < kChunks; kc++)
        {
            unsigned a0, a1, a2, a3;
            fa_ldsm_x4(a0, a1, a2, a3, Qs + fa_off<D>(qRow, kc * 16 + qCol));
            #pragma unroll
            for (unsigned np = 0; np < nTiles / 2; np++)
            {
                unsigned b0, b1, b2, b3;
                fa_ldsm_x4(b0, b1, b2, b3, Kst + fa_off<D>(np * 16 + kRow, kc * 16 + kCol));
                fa_mma(s[2 * np][0], s[2 * np][1], s[2 * np][2], s[2 * np][3], a0, a1, a2, a3, b0, b1);
                fa_mma(s[2 * np + 1][0], s[2 * np + 1][1], s[2 * np + 1][2], s[2 * np + 1][3], a0, a1, a2, a3, b2, b3);
            }
        }

        // ── Online softmax (log2 domain). Lane owns rows g and g+8, columns nt·8 + 2·t4 + {0,1}. ──
        float max0 = FA_NEG_INF, max1 = FA_NEG_INF;
        #pragma unroll
        for (unsigned nt = 0; nt < nTiles; nt++)
        {
            const unsigned c = nt * 8 + t4 * 2;
            s[nt][0] = c < cur ? s[nt][0] * scaleLog2 : FA_NEG_INF;
            s[nt][1] = c + 1 < cur ? s[nt][1] * scaleLog2 : FA_NEG_INF;
            s[nt][2] = c < cur ? s[nt][2] * scaleLog2 : FA_NEG_INF;
            s[nt][3] = c + 1 < cur ? s[nt][3] * scaleLog2 : FA_NEG_INF;
            max0 = fmaxf(max0, fmaxf(s[nt][0], s[nt][1]));
            max1 = fmaxf(max1, fmaxf(s[nt][2], s[nt][3]));
        }
        #pragma unroll
        for (unsigned off = 1; off <= 2; off <<= 1)
        {
            max0 = fmaxf(max0, __shfl_xor_sync(0xffffffffu, max0, off));
            max1 = fmaxf(max1, __shfl_xor_sync(0xffffffffu, max1, off));
        }
        const float mNew0 = fmaxf(m0, max0), mNew1 = fmaxf(m1, max1);
        const float corr0 = mNew0 == FA_NEG_INF ? 1.0f : exp2f(m0 - mNew0);
        const float corr1 = mNew1 == FA_NEG_INF ? 1.0f : exp2f(m1 - mNew1);
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (unsigned nt = 0; nt < nTiles; nt++)
        {
            s[nt][0] = s[nt][0] == FA_NEG_INF ? 0.0f : exp2f(s[nt][0] - mNew0);
            s[nt][1] = s[nt][1] == FA_NEG_INF ? 0.0f : exp2f(s[nt][1] - mNew0);
            s[nt][2] = s[nt][2] == FA_NEG_INF ? 0.0f : exp2f(s[nt][2] - mNew1);
            s[nt][3] = s[nt][3] == FA_NEG_INF ? 0.0f : exp2f(s[nt][3] - mNew1);
            sum0 += s[nt][0] + s[nt][1];
            sum1 += s[nt][2] + s[nt][3];
        }
        #pragma unroll
        for (unsigned off = 1; off <= 2; off <<= 1)
        {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, off);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, off);
        }
        l0 = l0 * corr0 + sum0;
        l1 = l1 * corr1 + sum1;
        m0 = mNew0;
        m1 = mNew1;
        // corr is exactly 1 wherever the running max held (exp2f(0) == 1), which is most steps past the first few;
        // skip the 128-multiply rescale unless some row in the warp moved.
        if (__any_sync(0xffffffffu, corr0 != 1.0f || corr1 != 1.0f))
        {
            #pragma unroll
            for (unsigned dt = 0; dt < dTiles; dt++)
            {
                o[dt][0] *= corr0; o[dt][1] *= corr0;
                o[dt][2] *= corr1; o[dt][3] *= corr1;
            }
        }

        fa_wait<0>();
        __syncthreads();   // V(step) landed; every warp is past S(step), so K's buffer is free
        if (step + 1 < nSteps)
        {
            fa_load_tile<D, FA_BC>(Kst, Kh, sk.row, (step + 1) * FA_BC, Skv);
            fa_commit();
        }

        // ── O += P·V. The m16n8k16 C layout is the A layout, so P n-tiles {2kc, 2kc+1} are k-chunk kc as-is. ──
        #pragma unroll
        for (unsigned kc = 0; kc < pChunks; kc++)
        {
            const __half2 p00 = __floats2half2_rn(s[2 * kc][0], s[2 * kc][1]);
            const __half2 p01 = __floats2half2_rn(s[2 * kc][2], s[2 * kc][3]);
            const __half2 p10 = __floats2half2_rn(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            const __half2 p11 = __floats2half2_rn(s[2 * kc + 1][2], s[2 * kc + 1][3]);
            const unsigned a0 = *(const unsigned*)&p00, a1 = *(const unsigned*)&p01;
            const unsigned a2 = *(const unsigned*)&p10, a3 = *(const unsigned*)&p11;
            #pragma unroll
            for (unsigned dp = 0; dp < dTiles / 2; dp++)
            {
                unsigned b0, b1, b2, b3;
                fa_ldsm_x4_t(b0, b1, b2, b3, Vst + fa_off<D>(kc * 16 + vRow, dp * 16 + vCol));
                fa_mma(o[2 * dp][0], o[2 * dp][1], o[2 * dp][2], o[2 * dp][3], a0, a1, a2, a3, b0, b1);
                fa_mma(o[2 * dp + 1][0], o[2 * dp + 1][1], o[2 * dp + 1][2], o[2 * dp + 1][3], a0, a1, a2, a3, b2, b3);
            }
        }
    }

    // ── Epilogue: O / l, stored as half2 at the C-fragment positions. ──
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const unsigned r0 = q0 + warp * 16 + g, r1 = r0 + 8;
    #pragma unroll
    for (unsigned dt = 0; dt < dTiles; dt++)
    {
        const unsigned d = dt * 8 + t4 * 2;
        if (r0 < Sq) *(__half2*)(Oh + (unsigned long long)r0 * so.row + d) = __floats2half2_rn(o[dt][0] * inv0, o[dt][1] * inv0);
        if (r1 < Sq) *(__half2*)(Oh + (unsigned long long)r1 * so.row + d) = __floats2half2_rn(o[dt][2] * inv1, o[dt][3] * inv1);
    }
}

#define FA_ENTRY(D)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(FA_THREADS, 1) flash_attn_f16_d##D(                                  \
        __half* __restrict__ O, const __half* __restrict__ Q, const __half* __restrict__ K,                           \
        const __half* __restrict__ V, FaStrides so, FaStrides sq, FaStrides sk, FaStrides sv,                        \
        unsigned int Sq, unsigned int Skv, float scaleLog2)                                                           \
    {                                                                                                                 \
        flash_attn_f16_core<D>(O, Q, K, V, so, sq, sk, sv, Sq, Skv, scaleLog2);                                       \
    }

FA_ENTRY(64)
FA_ENTRY(128)
FA_ENTRY(256)
