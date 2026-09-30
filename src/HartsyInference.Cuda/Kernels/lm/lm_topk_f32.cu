// Top-k over the last dimension of an F32 matrix by radix select, one row per block.
//
//   input [rows,n] F32, validLengths [rows] I32 (optional)  ->  values [rows,k] F32, indices [rows,k] I32
//
// Each row keeps the k largest of its first validLength entries. Ties go to the lowest index, NaN ranks below
// every number, and -0 equals +0, all matching TopKReference. The result is value-descending, or index-ascending
// when sortByIndex is set. Slots past the valid length hold index -1 and value -inf.
//
// How: keys are order-preserving uint32 images of the floats. Four 8-bit histogram passes find the k-th largest
// key T (shared-memory integer atomics only count, so the result is order independent). Every key above T is in
// the answer, and so are the first few keys equal to T by index. A chunked ballot scan compacts exactly those
// entries in index order, then a shared-memory bitonic sort orders them by (value desc, index asc) or by index.
//
// Launch: grid = rows, block = 256, k <= 2048 (the sort buffer is 16 KB of static shared memory).

#define NEG_INF __int_as_float(0xff800000)
#define TOPK_THREADS 256
#define TOPK_MAX_K 2048

__device__ __forceinline__ unsigned int topk_key(float v)
{
    if (v != v) return 0u;
    unsigned int b = __float_as_uint(v);
    if ((b & 0x7fffffffu) == 0u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(TOPK_THREADS) lm_topk_f32(
    float* __restrict__ values,
    int* __restrict__ indices,
    const float* __restrict__ input,
    const int* __restrict__ validLengths,
    int n,
    int k,
    int sortByIndex)
{
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long sortBuf[TOPK_MAX_K];
    __shared__ int warpCount[TOPK_THREADS / 32];
    __shared__ unsigned int prefix, threshold;
    __shared__ int remaining, eqNeeded, taken;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x = input + (long long)row * n;
    int len = validLengths == nullptr ? n : validLengths[row];
    len = len < 0 ? 0 : (len > n ? n : len);
    const int m = k < len ? k : len;

    if (m > 0)
    {
        if (tid == 0) { prefix = 0u; remaining = m; }
        __syncthreads();
        for (int pass = 0; pass < 4; pass++)
        {
            const int shift = 24 - 8 * pass;
            for (int i = tid; i < 256; i += TOPK_THREADS) hist[i] = 0u;
            __syncthreads();
            const unsigned int want = prefix;
            for (int i = tid; i < len; i += TOPK_THREADS)
            {
                const unsigned int key = topk_key(x[i]);
                if (pass == 0 || (key >> (shift + 8)) == want) atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (tid == 0)
            {
                int rem = remaining;
                int bin = 255;
                for (; bin > 0; bin--)
                {
                    if ((int)hist[bin] >= rem) break;
                    rem -= (int)hist[bin];
                }
                remaining = rem;
                prefix = (want << 8) | (unsigned int)bin;
            }
            __syncthreads();
        }
        if (tid == 0) { threshold = prefix; eqNeeded = remaining; taken = 0; }
        __syncthreads();

        const int lane = tid & 31, warp = tid >> 5;
        int eqSeen = 0;
        for (int start = 0; start < len; start += TOPK_THREADS)
        {
            const int i = start + tid;
            unsigned int key = 0u;
            bool gt = false, eq = false;
            if (i < len)
            {
                key = topk_key(x[i]);
                gt = key > threshold;
                eq = key == threshold;
            }
            const unsigned int eqBallot = __ballot_sync(0xffffffffu, eq);
            if (lane == 0) warpCount[warp] = __popc(eqBallot);
            __syncthreads();
            int warpEqOffset = 0, chunkEq = 0;
            for (int w = 0; w < TOPK_THREADS / 32; w++)
            {
                if (w < warp) warpEqOffset += warpCount[w];
                chunkEq += warpCount[w];
            }
            const int eqRank = eqSeen + warpEqOffset + __popc(eqBallot & ((1u << lane) - 1u));
            const bool take = gt || (eq && eqRank < eqNeeded);
            __syncthreads();

            const unsigned int takeBallot = __ballot_sync(0xffffffffu, take);
            if (lane == 0) warpCount[warp] = __popc(takeBallot);
            __syncthreads();
            int warpTakeOffset = 0, chunkTake = 0;
            for (int w = 0; w < TOPK_THREADS / 32; w++)
            {
                if (w < warp) warpTakeOffset += warpCount[w];
                chunkTake += warpCount[w];
            }
            if (take)
            {
                const int slot = taken + warpTakeOffset + __popc(takeBallot & ((1u << lane) - 1u));
                sortBuf[slot] = sortByIndex ? (unsigned long long)i
                                            : (((unsigned long long)(~key)) << 32) | (unsigned long long)i;
            }
            eqSeen += chunkEq;
            __syncthreads();
            if (tid == 0) taken += chunkTake;
            __syncthreads();
        }

        int size = 1;
        while (size < m) size <<= 1;
        for (int i = m + tid; i < size; i += TOPK_THREADS) sortBuf[i] = 0xffffffffffffffffull;
        __syncthreads();
        for (int span = 2; span <= size; span <<= 1)
        {
            for (int step = span >> 1; step > 0; step >>= 1)
            {
                for (int i = tid; i < size; i += TOPK_THREADS)
                {
                    const int j = i ^ step;
                    if (j > i)
                    {
                        const bool up = (i & span) == 0;
                        const unsigned long long a = sortBuf[i], b = sortBuf[j];
                        if ((a > b) == up) { sortBuf[i] = b; sortBuf[j] = a; }
                    }
                }
                __syncthreads();
            }
        }
    }

    for (int j = tid; j < k; j += TOPK_THREADS)
    {
        if (j < m)
        {
            const int idx = (int)(sortBuf[j] & 0xffffffffull);
            indices[(long long)row * k + j] = idx;
            values[(long long)row * k + j] = x[idx];
        }
        else
        {
            indices[(long long)row * k + j] = -1;
            values[(long long)row * k + j] = NEG_INF;
        }
    }
}
