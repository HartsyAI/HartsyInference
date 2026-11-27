// 3D block-sparse attention (IBackend.BlockSparseAttention): BlockSparseAttentionReference (Core) on the device. Operands are
// head-major [B, H, S, D] in (T, H, W) raster order, F32 or BF16. A BF16 value is widened to F32, exactly, as it is loaded;
// every sum, score and weight is F32.
//
//   1. bsa_means_<dtype>   per (block, head, batch): the mean row of one block of queries or of keys.
//   2. bsa_select          per (query block, head, batch): the key blocks it reads, ascending.
//   3. bsa_attend_<dtype>_d<D>
//                          per (query tile, head, batch): one softmax over the keys of the selected blocks, kept online
//                          (running maximum and sum) over chunks of key rows.
//
// Stages 1 and 2 decide which blocks are read, and that is a comparison: a score a rounding apart from the reference's can
// take another block. So they use the round-to-nearest intrinsics (__fadd_rn, __fmul_rn, __fdiv_rn), which the compiler
// does not contract into fused multiply-adds, in the reference's order: a mean over the block's rows in the block's raster
// order, a score over channels ascending. Their results are the reference's bit for bit, and so is the selection. Stage 3 is
// ordinary F32 arithmetic and is compared by tolerance.
//
// Build: ../attention/build.sh (nvrtc).

#include <cuda_bf16.h>

#define BSA_NEG_INF __int_as_float(0xff800000)
#define BSA_MAX_DIM 128

// Query rows and key rows of one stage-3 tile, and the query rows each thread accumulates.
#define BSA_QT 64
#define BSA_KT 64
#define BSA_QPT 4

__device__ __forceinline__ float bsa_load(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float bsa_load(const __nv_bfloat16* p, size_t i) { return __bfloat162float(p[i]); }

// The raster row of token `offset` of block `block` in a grid H tokens high and W wide, cut into bt x bh x bw blocks.
// Blocks and the tokens inside one are numbered in raster order (BlockSparseAttentionOptions.QueryRow / KeyRow).
__device__ __forceinline__ int bsa_row(int block, int offset, int H, int W, int bt, int bh, int bw)
{
    const int blocksHigh = H / bh, blocksWide = W / bw;
    const int frame = block / (blocksHigh * blocksWide) * bt + offset / (bh * bw);
    const int row = block / blocksWide % blocksHigh * bh + offset / bw % bh;
    const int column = block % blocksWide * bw + offset % bw;
    return (frame * H + row) * W + column;
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Block means. Grid (blocks, heads, batch), block 128: thread d owns channel d.
template <typename T>
__device__ __forceinline__ void bsa_means_impl(float* __restrict__ means, const T* __restrict__ x, int tokens, int D, int heads,
    int blocks, int gridH, int gridW, int bt, int bh, int bw)
{
    const int n = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const size_t head = (size_t)b * heads + h;
    const T* rows = x + head * tokens * D;
    const int length = bt * bh * bw;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int o = 0; o < length; ++o)
            sum = __fadd_rn(sum, bsa_load(rows, (size_t)bsa_row(n, o, gridH, gridW, bt, bh, bw) * D + d));
        means[(head * blocks + n) * D + d] = __fdiv_rn(sum, (float)length);
    }
}

extern "C" __global__ void bsa_means_f32(float* means, const float* x, int tokens, int D, int heads, int blocks, int gridH,
    int gridW, int bt, int bh, int bw)
{
    bsa_means_impl(means, x, tokens, D, heads, blocks, gridH, gridW, bt, bh, bw);
}

extern "C" __global__ void bsa_means_bf16(float* means, const __nv_bfloat16* x, int tokens, int D, int heads, int blocks,
    int gridH, int gridW, int bt, int bh, int bw)
{
    bsa_means_impl(means, x, tokens, D, heads, blocks, gridH, gridW, bt, bh, bw);
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Selection.

// An unsigned key that orders as the reference ranks scores: NaN above every number, then by value, -0 equal to +0.
__device__ __forceinline__ unsigned int bsa_rank_key(float score)
{
    if (score != score) return 0xffffffffu;
    if (score == 0.0f) return 0x80000000u;
    const unsigned int bits = __float_as_uint(score);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// Grid (query blocks, heads, batch), block 256, dynamic shared memory of one key per key block. Takes `chosen` >= 1 blocks:
// the threshold is the chosen-th largest key, found a bit at a time; every block above it is read, and as many of the blocks
// equal to it as are still wanted, lowest index first.
extern "C" __global__ void bsa_select(int* __restrict__ selected, const float* __restrict__ queryMeans,
    const float* __restrict__ keyMeans, int D, int heads, int queryBlocks, int keyBlocks, int chosen)
{
    extern __shared__ unsigned int keys[];
    __shared__ float mean[BSA_MAX_DIM];
    __shared__ int warpCounts[8];
    const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const size_t head = (size_t)b * heads + h;

    for (int d = tid; d < D; d += blockDim.x) mean[d] = queryMeans[(head * queryBlocks + qb) * D + d];
    __syncthreads();
    for (int n = tid; n < keyBlocks; n += blockDim.x) {
        const float* k = keyMeans + (head * keyBlocks + n) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot = __fadd_rn(dot, __fmul_rn(mean[d], k[d]));
        keys[n] = bsa_rank_key(dot);
    }
    __syncthreads();

    unsigned int threshold = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned int candidate = threshold | (1u << bit);
        int count = 0;
        for (int n = tid; n < keyBlocks; n += blockDim.x) count += keys[n] >= candidate ? 1 : 0;
        for (int offset = 16; offset > 0; offset >>= 1) count += __shfl_down_sync(0xffffffffu, count, offset);
        if (lane == 0) warpCounts[warp] = count;
        __syncthreads();
        int total = 0;
        for (int w = 0; w < 8; ++w) total += warpCounts[w];
        __syncthreads();
        if (total >= chosen) threshold = candidate;
    }

    if (tid != 0) return;
    int ties = chosen;
    for (int n = 0; n < keyBlocks; ++n) ties -= keys[n] > threshold ? 1 : 0;
    int* out = selected + (head * queryBlocks + qb) * chosen;
    for (int n = 0, at = 0; n < keyBlocks && at < chosen; ++n) {
        if (keys[n] > threshold) out[at++] = n;
        else if (keys[n] == threshold && ties > 0) { out[at++] = n; --ties; }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Attention over the selected blocks. Grid (query blocks x tiles per block, heads, batch); LANES threads share a group of
// BSA_QPT query rows, each scoring BSA_KT / LANES key rows of a chunk and accumulating D / LANES channels. Dynamic shared
// memory: the query tile, one chunk of K and then of V in the same rows, the chunk's weights and the chunk's row numbers
// (CudaKernels.BlockSparseAttentionSharedBytes). Rows are padded by one float against bank conflicts.
template <typename T, int HD, int LANES>
__device__ __forceinline__ void bsa_attend_impl(void* __restrict__ output, const T* __restrict__ Q, const T* __restrict__ K,
    const T* __restrict__ V, const int* __restrict__ selected, int queryTokens, int keyTokens, int heads, int queryBlocks,
    int chosen, int qGridH, int qGridW, int qbt, int qbh, int qbw, int kGridH, int kGridW, int kbt, int kbh, int kbw,
    float scale, int outputBf16)
{
    const int GROUPS = BSA_QT / BSA_QPT;
    const int NT = GROUPS * LANES;
    const int PPT = BSA_KT / LANES;
    const int DPT = HD / LANES;
    const int STRIDE = HD + 1;
    const int WSTRIDE = BSA_KT + 1;

    extern __shared__ float smem[];
    float* qs = smem;
    float* ks = qs + BSA_QT * STRIDE;
    float* weights = ks + BSA_KT * STRIDE;
    int* keyRows = (int*)(weights + BSA_QT * WSTRIDE);

    const int queryLength = qbt * qbh * qbw, keyLength = kbt * kbh * kbw;
    const int tiles = (queryLength + BSA_QT - 1) / BSA_QT;
    const int qb = blockIdx.x / tiles, tile = blockIdx.x % tiles;
    const int h = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x;
    const size_t head = (size_t)b * heads + h;
    const T* queries = Q + head * queryTokens * HD;
    const T* keys = K + head * keyTokens * HD;
    const T* values = V + head * keyTokens * HD;
    const int* blocks = selected + (head * queryBlocks + qb) * chosen;

    for (int i = tid; i < BSA_QT * HD; i += NT) {
        const int row = i / HD, d = i - row * HD;
        const int offset = tile * BSA_QT + row;
        qs[row * STRIDE + d] = offset < queryLength
            ? bsa_load(queries, (size_t)bsa_row(qb, offset, qGridH, qGridW, qbt, qbh, qbw) * HD + d) : 0.0f;
    }

    const int group = tid / LANES, lane = tid % LANES;
    const int qBase = group * BSA_QPT, pBase = lane * PPT, dBase = lane * DPT;

    // The softmax state lives in registers, replicated over the LANES threads of a group: every update is a reduction over
    // exactly those lanes, so the copies cannot drift.
    float acc[BSA_QPT][DPT], runMax[BSA_QPT], runSum[BSA_QPT];
#pragma unroll
    for (int i = 0; i < BSA_QPT; ++i) {
        runMax[i] = BSA_NEG_INF;
        runSum[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < DPT; ++j) acc[i][j] = 0.0f;
    }

    const int total = chosen * keyLength;
    for (int start = 0; start < total; start += BSA_KT) {
        __syncthreads();
        for (int p = tid; p < BSA_KT; p += NT) {
            const int at = start + p;
            keyRows[p] = at < total
                ? bsa_row(blocks[at / keyLength], at % keyLength, kGridH, kGridW, kbt, kbh, kbw) : -1;
        }
        __syncthreads();
        for (int i = tid; i < BSA_KT * HD; i += NT) {
            const int p = i / HD, d = i - p * HD;
            ks[p * STRIDE + d] = keyRows[p] >= 0 ? bsa_load(keys, (size_t)keyRows[p] * HD + d) : 0.0f;
        }
        __syncthreads();

        float score[BSA_QPT][PPT];
#pragma unroll
        for (int i = 0; i < BSA_QPT; ++i)
#pragma unroll
            for (int j = 0; j < PPT; ++j) score[i][j] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < HD; ++d) {
            float qv[BSA_QPT], kv[PPT];
#pragma unroll
            for (int i = 0; i < BSA_QPT; ++i) qv[i] = qs[(qBase + i) * STRIDE + d];
#pragma unroll
            for (int j = 0; j < PPT; ++j) kv[j] = ks[(pBase + j) * STRIDE + d];
#pragma unroll
            for (int i = 0; i < BSA_QPT; ++i)
#pragma unroll
                for (int j = 0; j < PPT; ++j) score[i][j] += qv[i] * kv[j];
        }

#pragma unroll
        for (int i = 0; i < BSA_QPT; ++i) {
            // A row past the keys is no key: negative infinity. The comparison skips a NaN score, which reaches the sum
            // through its weight below and so reaches the output.
            float chunkMax = BSA_NEG_INF;
#pragma unroll
            for (int j = 0; j < PPT; ++j) {
                score[i][j] = keyRows[pBase + j] >= 0 ? score[i][j] * scale : BSA_NEG_INF;
                if (score[i][j] > chunkMax) chunkMax = score[i][j];
            }
#pragma unroll
            for (int offset = LANES >> 1; offset > 0; offset >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, chunkMax, offset);
                if (other > chunkMax) chunkMax = other;
            }
            const float newMax = chunkMax > runMax[i] ? chunkMax : runMax[i];
            const float rescale = newMax == runMax[i] ? 1.0f : expf(runMax[i] - newMax);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PPT; ++j) {
                const float weight = score[i][j] == BSA_NEG_INF ? 0.0f : expf(score[i][j] - newMax);
                weights[(qBase + i) * WSTRIDE + pBase + j] = weight;
                sum += weight;
            }
#pragma unroll
            for (int offset = LANES >> 1; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            runMax[i] = newMax;
            runSum[i] = runSum[i] * rescale + sum;
#pragma unroll
            for (int j = 0; j < DPT; ++j) acc[i][j] *= rescale;
        }
        __syncthreads();

        for (int i = tid; i < BSA_KT * HD; i += NT) {
            const int p = i / HD, d = i - p * HD;
            ks[p * STRIDE + d] = keyRows[p] >= 0 ? bsa_load(values, (size_t)keyRows[p] * HD + d) : 0.0f;
        }
        __syncthreads();

        for (int p = 0; p < BSA_KT; ++p) {
            if (keyRows[p] < 0) break;
            float w[BSA_QPT], vv[DPT];
#pragma unroll
            for (int i = 0; i < BSA_QPT; ++i) w[i] = weights[(qBase + i) * WSTRIDE + p];
#pragma unroll
            for (int j = 0; j < DPT; ++j) vv[j] = ks[p * STRIDE + dBase + j];
#pragma unroll
            for (int i = 0; i < BSA_QPT; ++i)
#pragma unroll
                for (int j = 0; j < DPT; ++j) acc[i][j] += w[i] * vv[j];
        }
    }

#pragma unroll
    for (int i = 0; i < BSA_QPT; ++i) {
        const int offset = tile * BSA_QT + qBase + i;
        if (offset >= queryLength) continue;
        const size_t at = (head * queryTokens + bsa_row(qb, offset, qGridH, qGridW, qbt, qbh, qbw)) * HD + dBase;
        // A query with no key is zero; a NaN sum is not zero and stays NaN.
        const bool empty = runSum[i] == 0.0f;
        const float inverse = empty ? 0.0f : 1.0f / runSum[i];
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const float result = empty ? 0.0f : acc[i][j] * inverse;
            if (outputBf16) ((__nv_bfloat16*)output)[at + j] = __float2bfloat16(result);
            else ((float*)output)[at + j] = result;
        }
    }
}

#define BSA_LANES 16
#define BSA_ATTEND(NAME, TYPE, HD)                                                                                          \
    extern "C" __global__ __launch_bounds__(BSA_QT / BSA_QPT * BSA_LANES) void NAME(                                       \
        void* output, const TYPE* Q, const TYPE* K, const TYPE* V, const int* selected, int queryTokens, int keyTokens,     \
        int heads, int queryBlocks, int chosen, int qGridH, int qGridW, int qbt, int qbh, int qbw, int kGridH, int kGridW, \
        int kbt, int kbh, int kbw, float scale, int outputBf16)                                                            \
    {                                                                                                                       \
        bsa_attend_impl<TYPE, HD, BSA_LANES>(output, Q, K, V, selected, queryTokens, keyTokens, heads, queryBlocks, chosen, \
            qGridH, qGridW, qbt, qbh, qbw, kGridH, kGridW, kbt, kbh, kbw, scale, outputBf16);                               \
    }

BSA_ATTEND(bsa_attend_f32_d16, float, 16)
BSA_ATTEND(bsa_attend_f32_d32, float, 32)
BSA_ATTEND(bsa_attend_f32_d64, float, 64)
BSA_ATTEND(bsa_attend_f32_d128, float, 128)
BSA_ATTEND(bsa_attend_bf16_d16, __nv_bfloat16, 16)
BSA_ATTEND(bsa_attend_bf16_d32, __nv_bfloat16, 32)
BSA_ATTEND(bsa_attend_bf16_d64, __nv_bfloat16, 64)
BSA_ATTEND(bsa_attend_bf16_d128, __nv_bfloat16, 128)
