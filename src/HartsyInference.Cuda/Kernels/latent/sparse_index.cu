// Sparse-attention index building (IBackend.SparseTopK, IBackend.MergeWindowTopK; host references SparseTopKReference and
// WindowTopKMergeReference).
//
// sparse_topk_f32
//   One block per score row. Four 8-bit radix passes over an order-preserving key find the exact key T of the k-th best
//   score and how many scores equal to T it takes; one pass in column order then writes every score above T and the
//   lowest-column ties, which leaves the row in ascending column order. NaN ranks below every number and -0 equals +0,
//   as float.CompareTo orders them; columns at or past the row's limit become -1. Exact, so it matches the host bit for bit.
//
// merge_window_topk_i32
//   One thread per output entry: [topk[:n] | seq-w .. seq-1 | -1 ...] with seq = position + 1,
//   n = min(max(seq - window, 0), topkCols) and w = min(seq, window).

#define TOPK_THREADS 512
#define TOPK_WARPS (TOPK_THREADS / 32)

// Larger key = better score; NaN is 0, below -inf.
__device__ __forceinline__ unsigned int topk_key(float v)
{
    if (isnan(v)) return 0u;
    unsigned int b = v == 0.0f ? 0u : __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Exclusive prefix sum of v over the block; *total receives the block sum. Every thread must call it.
__device__ __forceinline__ int topk_block_scan(int v, int* warpSums, int* total)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1)
    {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warpSums[warp] = x;
    __syncthreads();
    if (warp == 0)
    {
        int s = lane < TOPK_WARPS ? warpSums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1)
        {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < TOPK_WARPS) warpSums[lane] = s;
    }
    __syncthreads();
    const int before = warp > 0 ? warpSums[warp - 1] : 0;
    *total = warpSums[TOPK_WARPS - 1];
    __syncthreads();
    return before + x - v;
}

extern "C" __global__ void __launch_bounds__(TOPK_THREADS)
sparse_topk_f32(const float* __restrict__ scores, const int* __restrict__ limits, int* __restrict__ out, int cols, int k, int offset)
{
    __shared__ unsigned int hist[256];
    __shared__ int warpSums[TOPK_WARPS];
    __shared__ unsigned int sPrefix;
    __shared__ int sRemaining;
    const int row = blockIdx.x;
    const float* s = scores + (long long)row * cols;
    int* dst = out + (long long)row * k;
    const int limit = limits[row];

    unsigned int prefix = 0u, mask = 0u;
    int remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        for (int b = threadIdx.x; b < 256; b += TOPK_THREADS) hist[b] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < cols; i += TOPK_THREADS)
        {
            const unsigned int key = topk_key(s[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int above = 0, bin = 0;
            for (int b = 255; b >= 0; b--)
            {
                const int c = (int)hist[b];
                if (above + c >= remaining) { bin = b; break; }
                above += c;
            }
            sPrefix = prefix | ((unsigned int)bin << shift);
            sRemaining = remaining - above;
        }
        __syncthreads();
        prefix = sPrefix;
        remaining = sRemaining;
        mask |= 255u << shift;
        __syncthreads();
    }

    // prefix is the k-th best key; remaining of the scores equal to it are taken, lowest columns first.
    int taken = 0, eqSeen = 0;
    for (int start = 0; start < cols && taken < k; start += TOPK_THREADS)
    {
        const int i = start + threadIdx.x;
        const unsigned int key = i < cols ? topk_key(s[i]) : 0u;
        const int gt = i < cols && key > prefix;
        const int eq = i < cols && key == prefix;
        int eqTotal, selTotal;
        const int eqRank = eqSeen + topk_block_scan(eq, warpSums, &eqTotal);
        const int sel = gt || (eq && eqRank < remaining);
        const int pos = taken + topk_block_scan(sel, warpSums, &selTotal);
        if (sel) dst[pos] = i < limit ? i + offset : -1;
        eqSeen += eqTotal;
        taken += selTotal;
    }
}

extern "C" __global__ void merge_window_topk_i32(const int* __restrict__ topk, const int* __restrict__ positions, int* __restrict__ rows,
                                                 long long total, int topkCols, int window, int width)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const long long t = idx / width;
    const int i = (int)(idx - t * width);
    const int seq = positions[t] + 1;
    const int n = min(max(seq - window, 0), topkCols);
    const int w = min(seq, window);
    rows[idx] = i < n ? topk[t * topkCols + i] : i < n + w ? seq - w + (i - n) : -1;
}
