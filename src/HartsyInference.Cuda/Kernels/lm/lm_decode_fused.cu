// Graph-decode fusions (t = 1, B = 1). Each kernel here replaces a short chain of existing launches and is BIT-IDENTICAL to that
// chain: the reductions, the expressions and their evaluation order are the originals', only the launch boundaries move.
//
//  lm_rmsnorm_q8_1_wide_kK / lm_add_rmsnorm_q8_1_wide_kK  (K = row width / 256, 2..32)
//      RMSNorm + Q8_1 sidecar with the reference reduction (the per-thread partials and tree of lm_norm_q8_fast.cu, themselves
//      bit-identical to lm_f32.cu's shared-memory kernels) on up to 1024 threads: the row is staged in shared memory, the
//      normalize + quantize pass runs on 32 warps instead of 8, and every multiple of 256 is served (lm_norm_q8_fast.cu only
//      instantiates powers of two, so hidden 2560, 3072, 5120 ... ran the generic kernel at about twice the time).
//  moe_combine_add_rmsnorm_q8_1_kK
//      moe_combine_slots_f32 + the residual add + the next add-RMSNorm Q8 (three launches -> one): the producer computes each
//      combined element exactly as moe_combine_slots does, then the wide norm above runs.
//  lm_qknorm_full_rope_scatter_f32 / _f16kv
//      Full-width QK-RMSNorm (OLMoE: one norm over all q heads, one over all k heads) + RoPE + KV scatter: replaces two or
//      three SliceLastDim launches, two dit_rmsnorm_f32 launches and the rope-scatter. Every head block recomputes its
//      section's sum of squares with dit_rmsnorm_f32's own strided partial and tree at blockDim 256, so invRms is the bits
//      the norm kernel computes; the section (8 KB for OLMoE) is re-read from L2 by each head.
//  lm_flash_attn_f32_combine_q8_1
//      lm_flash_attn_f32_combine plus the Q8_1 sidecar of its output, so the o-projection's dp4a GEMV skips its quantize
//      launch. Each warp owns 32 consecutive elements of one head (D is a multiple of 32), i.e. one aligned 32-block.
//
// Build: ./build.sh (nvcc -ptx -arch=sm_80).

#include <cuda_fp16.h>
#include "lm_norm_q8_fast.cu"   // FAST_THREADS, fast_quantize_block (its own kernels come along unused)

// ---------------------------------------------------------------------------------------------------------------------------
// Wide RMSNorm + Q8_1 (one row per block). The row is produced by WIDE_NT threads into shared memory (input, a + b, or a + MoE
// combine), then threads 0..255 form exactly lm_norm_q8_fast.cu's per-thread partials (element t + 256 k, k ascending) and
// fast_tree_sum's tree, so invRms has the reference kernels' bits. Launch: grid = rows, block = WIDE_NT(K), static shared memory.

#define WIDE_NT(K) ((K) < 4 ? 256 * (K) : 1024)

// Producers: the residual value of element i of the row.
struct NormIn { const float* x; __device__ float operator()(unsigned int i) const { return x[i]; } };
struct AddIn  { const float* a; const float* b; __device__ float operator()(unsigned int i) const { return a[i] + b[i]; } };
struct MoeIn {
    const float* a; const float* slotOut; const float* topkWeight; const float* shared; float gate; int hasGate; int topk; unsigned int hidden;
    __device__ float operator()(unsigned int i) const
    {
        // Verbatim moe_combine_slots_f32 for one token (the gate's sigmoid is hoisted; same expression on the same logit).
        float acc = 0.0f;
        if (shared != nullptr) {
            float s = shared[i];
            if (hasGate) s *= gate;
            acc = s;
        }
        #pragma unroll 8
        for (int j = 0; j < topk; ++j)
            acc += topkWeight[j] * slotOut[(size_t)j * hidden + i];
        return a[i] + acc;
    }
};

template <int K, typename Producer>
__device__ __forceinline__ void wide_norm_q8(
    float* __restrict__ residOut, float* __restrict__ normOut, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs,
    Producer produce, const float* __restrict__ weight, unsigned int normDim, float eps)
{
    constexpr unsigned int NT = WIDE_NT(K);
    __shared__ float sx[FAST_THREADS * K];
    __shared__ float sdata[FAST_THREADS];
    __shared__ float sInv;
    const unsigned int t = threadIdx.x, lane = t & 31u, warp = t >> 5;

    #pragma unroll
    for (unsigned int i = t; i < FAST_THREADS * K; i += NT) {
        const float v = produce(i);
        if (residOut != nullptr) residOut[i] = v;
        sx[i] = v;
    }
    __syncthreads();

    if (t < FAST_THREADS) {
        float partial = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const float v = sx[t + FAST_THREADS * k];
            partial += v * v;
        }
        sdata[t] = partial;
    }
    __syncthreads();
    if (warp == 0) {
        // fast_tree_sum's warp-0 stage, verbatim.
        const float a0 = sdata[lane],       a1 = sdata[32 + lane],  a2 = sdata[64 + lane],  a3 = sdata[96 + lane];
        const float a4 = sdata[128 + lane], a5 = sdata[160 + lane], a6 = sdata[192 + lane], a7 = sdata[224 + lane];
        const float b0 = a0 + a4, b1 = a1 + a5, b2 = a2 + a6, b3 = a3 + a7;
        const float c0 = b0 + b2, c1 = b1 + b3;
        float d = c0 + c1;
        d += __shfl_down_sync(0xffffffffu, d, 16);
        d += __shfl_down_sync(0xffffffffu, d, 8);
        d += __shfl_down_sync(0xffffffffu, d, 4);
        d += __shfl_down_sync(0xffffffffu, d, 2);
        d += __shfl_down_sync(0xffffffffu, d, 1);
        if (lane == 0) sInv = rsqrtf(d / (float)normDim + eps);
    }
    __syncthreads();
    const float invRms = sInv;

    // Warp w handles elements 32 w + NT j + lane: one aligned 32-block per (warp, j).
    #pragma unroll
    for (unsigned int i = t; i < FAST_THREADS * K; i += NT) {
        const float o = sx[i] * invRms * weight[i];
        normOut[i] = o;
        const unsigned int blk = i >> 5;
        fast_quantize_block(xq + (size_t)blk * 32, xd + blk, xs + blk, o, lane);
    }
}

#define WIDE(K) \
extern "C" __global__ void __launch_bounds__(WIDE_NT(K)) lm_rmsnorm_q8_1_wide_k##K( \
    float* __restrict__ output, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs, \
    const float* __restrict__ input, const float* __restrict__ weight, unsigned int normDim, unsigned int totalRows, float eps) \
{ \
    const size_t r = blockIdx.x; if (r >= totalRows) return; \
    wide_norm_q8<K>(nullptr, output + r * normDim, xq + r * normDim, xd + r * (normDim / 32u), xs + r * (normDim / 32u), \
                    NormIn{input + r * normDim}, weight, normDim, eps); \
} \
extern "C" __global__ void __launch_bounds__(WIDE_NT(K)) lm_add_rmsnorm_q8_1_wide_k##K( \
    float* __restrict__ residOut, float* __restrict__ normOut, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs, \
    const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ weight, unsigned int normDim, unsigned int totalRows, float eps) \
{ \
    const size_t r = blockIdx.x; if (r >= totalRows) return; \
    wide_norm_q8<K>(residOut + r * normDim, normOut + r * normDim, xq + r * normDim, xd + r * (normDim / 32u), xs + r * (normDim / 32u), \
                    AddIn{a + r * normDim, b + r * normDim}, weight, normDim, eps); \
} \
extern "C" __global__ void __launch_bounds__(WIDE_NT(K)) moe_combine_add_rmsnorm_q8_1_k##K( \
    float* __restrict__ residOut, float* __restrict__ normOut, signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs, \
    const float* __restrict__ a, const float* __restrict__ slotOut, const float* __restrict__ topkWeight, \
    const float* __restrict__ shared, const float* __restrict__ sharedGateLogit, int topk, \
    const float* __restrict__ weight, unsigned int normDim, float eps) \
{ \
    const int hasGate = sharedGateLogit != nullptr; \
    const float gate = hasGate ? 1.0f / (1.0f + expf(-sharedGateLogit[0])) : 1.0f; \
    wide_norm_q8<K>(residOut, normOut, xq, xd, xs, MoeIn{a, slotOut, topkWeight, shared, gate, hasGate, topk, normDim}, weight, normDim, eps); \
}

WIDE(2)  WIDE(3)  WIDE(4)  WIDE(5)  WIDE(6)  WIDE(7)  WIDE(8)  WIDE(9)  WIDE(10) WIDE(11) WIDE(12)
WIDE(13) WIDE(14) WIDE(15) WIDE(16) WIDE(17) WIDE(18) WIDE(19) WIDE(20) WIDE(21) WIDE(22) WIDE(23)
WIDE(24) WIDE(25) WIDE(26) WIDE(27) WIDE(28) WIDE(29) WIDE(30) WIDE(31) WIDE(32)

// ---------------------------------------------------------------------------------------------------------------------------
// Full-width QK-norm + RoPE + KV scatter. Block h: h < nq -> q head h (norm over all nq*D q values, rope -> qOut);
// h < nq + nkv -> k head (norm over all nkv*D k values, rope -> kCache slot); else v head (copy -> vCache slot).
// Launch: grid = nq + 2 * nkv, block = 256 (dit_rmsnorm_f32's production geometry), shared = 256 floats.

__device__ __forceinline__ void kv_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void kv_store(__half* p, float v) { *p = __float2half_rn(v); }

template <typename KvT>
__device__ __forceinline__ void qknorm_full_rope_scatter(
    float* __restrict__ qOut, KvT* __restrict__ kCache, KvT* __restrict__ vCache,
    const float* __restrict__ qIn, const float* __restrict__ kIn, const float* __restrict__ vIn,
    const float* __restrict__ qNormW, const float* __restrict__ kNormW,
    const float* __restrict__ cosTable, const float* __restrict__ sinTable,
    unsigned int nq, unsigned int nkv, unsigned int headDim, unsigned int rotaryDim, int interleaved, float eps,
    unsigned int maxSeq, const int* __restrict__ devicePos)
{
    extern __shared__ float sdata_qfs[];
    const unsigned int h = blockIdx.x;
    if (h >= nq + 2u * nkv) return;
    const int pos = devicePos[1];

    if (h >= nq + nkv) {
        const unsigned int vh = h - nq - nkv;
        const float* src = vIn + (size_t)vh * headDim;
        KvT* dst = vCache + (((unsigned long long)vh * maxSeq) + (unsigned int)pos) * headDim;
        for (unsigned int i = threadIdx.x; i < headDim; i += blockDim.x) kv_store(dst + i, src[i]);
        return;
    }

    const bool isQ = h < nq;
    const unsigned int sh = isQ ? h : h - nq;
    const float* sec = isQ ? qIn : kIn;                    // the whole q (or k) section: the norm's row
    const unsigned int normDim = (isQ ? nq : nkv) * headDim;
    const float* w = isQ ? qNormW : kNormW;

    // dit_rmsnorm_f32 reduction, verbatim, over the section.
    float partial = 0.0f;
    for (unsigned int i = threadIdx.x; i < normDim; i += blockDim.x) {
        float v = sec[i];
        partial += v * v;
    }
    sdata_qfs[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_qfs[threadIdx.x] += sdata_qfs[threadIdx.x + s];
        __syncthreads();
    }
    const float invRms = rsqrtf(sdata_qfs[0] / (float)normDim + eps);

    const unsigned int hb = sh * headDim;                  // this head's first element within the section
    const size_t baseCs = (size_t)pos * headDim;
    for (unsigned int i = threadIdx.x; i < headDim; i += blockDim.x) {
        const float self = sec[hb + i] * invRms * w[hb + i];   // dit_rmsnorm_f32's output expression
        float outv;
        if (interleaved) {
            const unsigned int p = i >> 1, j = i ^ 1u;
            const float other = sec[hb + j] * invRms * w[hb + j];
            const float c = cosTable[baseCs + p], s = sinTable[baseCs + p];
            outv = (i & 1u) == 0u ? self * c - other * s : self * c + other * s;
        } else {
            const unsigned int rdim = (rotaryDim == 0u || rotaryDim > headDim) ? headDim : rotaryDim;
            const unsigned int half = rdim >> 1;
            if (i < half) {
                const unsigned int j = i + half;
                const float other = sec[hb + j] * invRms * w[hb + j];
                outv = self * cosTable[baseCs + i] - other * sinTable[baseCs + i];
            } else if (i < rdim) {
                const unsigned int j = i - half;
                const float other = sec[hb + j] * invRms * w[hb + j];
                outv = self * cosTable[baseCs + i] + other * sinTable[baseCs + i];
            } else {
                outv = self;
            }
        }
        if (isQ) qOut[hb + i] = outv;
        else kv_store(kCache + (((unsigned long long)sh * maxSeq) + (unsigned int)pos) * headDim + i, outv);
    }
}

#define QKFULL(SUFFIX, KVT) \
extern "C" __global__ void lm_qknorm_full_rope_scatter_##SUFFIX( \
    float* __restrict__ qOut, KVT* __restrict__ kCache, KVT* __restrict__ vCache, \
    const float* __restrict__ qIn, const float* __restrict__ kIn, const float* __restrict__ vIn, \
    const float* __restrict__ qNormW, const float* __restrict__ kNormW, \
    const float* __restrict__ cosTable, const float* __restrict__ sinTable, \
    unsigned int nq, unsigned int nkv, unsigned int headDim, unsigned int rotaryDim, int interleaved, float eps, \
    unsigned int maxSeq, const int* __restrict__ devicePos) \
{ qknorm_full_rope_scatter<KVT>(qOut, kCache, vCache, qIn, kIn, vIn, qNormW, kNormW, cosTable, sinTable, \
                                nq, nkv, headDim, rotaryDim, interleaved, eps, maxSeq, devicePos); }

QKFULL(f32, float)
QKFULL(f16kv, __half)

// ---------------------------------------------------------------------------------------------------------------------------
// lm_flash_attn_f32_combine (verbatim) + Q8_1 sidecar of out. One block per (b, head, row); blockDim = next pow2 >= max(D, 32).
// D % 32 == 0 (host-gated), so the warps with tid < D are full and each holds one aligned 32-block of the output.

#define INF_FC 3.402823466e+38f   // INF_F of flash_attn_f32_split.cu

extern "C" __global__ void lm_flash_attn_f32_combine_q8_1(
    float* __restrict__ out,
    signed char* __restrict__ xq, float* __restrict__ xd, float* __restrict__ xs,
    const float* __restrict__ partialM,
    const float* __restrict__ partialL,
    const float* __restrict__ partialAcc,
    unsigned int B, unsigned int Hq, unsigned int Tq, unsigned int D, unsigned int G)
{
    unsigned int idx = blockIdx.x;
    unsigned int total = B * Hq * Tq;
    if (idx >= total) return;
    unsigned int tid = threadIdx.x;

    float m = -INF_FC;
    for (unsigned int g = 0; g < G; g++) {
        float mg = partialM[idx * G + g];
        m = fmaxf(m, mg);
    }

    float l = 0.0f, acc = 0.0f;
    for (unsigned int g = 0; g < G; g++) {
        float mg = partialM[idx * G + g];
        float w = (mg <= -INF_FC) ? 0.0f : __expf(mg - m);
        l += partialL[idx * G + g] * w;
        if (tid < D) acc += partialAcc[(size_t)(idx * G + g) * D + tid] * w;
    }

    if (tid < D) {
        size_t qBase = (size_t)idx * D;
        const float o = l > 0.0f ? acc / l : 0.0f;
        out[qBase + tid] = o;
        const size_t blk = (qBase + tid) >> 5;
        fast_quantize_block(xq + blk * 32, xd + blk, xs + blk, o, tid & 31u);
    }
}
