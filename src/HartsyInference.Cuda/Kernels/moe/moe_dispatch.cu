// Deterministic, atomic-free MoE dispatch and combine.
//
// Dispatch turns the [T,k] expert ids into an expert-major row order in three launches:
//   count   : one block per expert counts the pairs that chose it
//   scan    : one thread turns the counts into exclusive offsets (E is small)
//   scatter : one block per expert walks the pairs in order and assigns each of its pairs the next row in its
//             segment with a warp-ballot ordered compaction, so rows inside an expert keep flat pair order; one
//             extra block marks dropped pairs (expert id out of range) and pads permutedToken past offsets[E]
// No atomics touch global memory, so the permutation is a pure function of the ids.
//
// Combine gathers each token's k expert rows: out[t] (+)= sum_j w[t,j] * expertOut[pairSlot[t,j]], j ascending,
// with IEEE multiply/add (no contraction) so it matches the CPU reference bit for bit.

#define DISPATCH_THREADS 256

extern "C" __global__ void __launch_bounds__(DISPATCH_THREADS) moe_dispatch_count_i32(
    int* __restrict__ counts,
    const int* __restrict__ topkIdx,
    int pairs)
{
    __shared__ int warpSum[DISPATCH_THREADS / 32];
    const int e = blockIdx.x;
    int c = 0;
    for (int p = threadIdx.x; p < pairs; p += DISPATCH_THREADS) c += topkIdx[p] == e ? 1 : 0;
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0) warpSum[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int total = 0;
        for (int w = 0; w < DISPATCH_THREADS / 32; w++) total += warpSum[w];
        counts[e] = total;
    }
}

extern "C" __global__ void moe_dispatch_scan_i32(
    int* __restrict__ offsets,
    const int* __restrict__ counts,
    int numExperts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int total = 0;
    for (int e = 0; e < numExperts; e++)
    {
        offsets[e] = total;
        total += counts[e];
    }
    offsets[numExperts] = total;
}

extern "C" __global__ void __launch_bounds__(DISPATCH_THREADS) moe_dispatch_scatter_i32(
    int* __restrict__ permutedToken,
    int* __restrict__ pairSlot,
    const int* __restrict__ topkIdx,
    const int* __restrict__ offsets,
    int pairs,
    int k,
    int numExperts)
{
    __shared__ int warpCount[DISPATCH_THREADS / 32];
    const int e = blockIdx.x;
    const int tid = threadIdx.x;

    if (e == numExperts)
    {
        const int total = offsets[numExperts];
        for (int p = tid; p < pairs; p += DISPATCH_THREADS)
        {
            int id = topkIdx[p];
            if (id < 0 || id >= numExperts) pairSlot[p] = -1;
            if (p >= total) permutedToken[p] = -1;
        }
        return;
    }

    const int base = offsets[e];
    const int lane = tid & 31, warp = tid >> 5;
    int running = 0;
    for (int start = 0; start < pairs; start += DISPATCH_THREADS)
    {
        const int p = start + tid;
        const bool flag = p < pairs && topkIdx[p] == e;
        const unsigned int ballot = __ballot_sync(0xffffffffu, flag);
        const int rank = __popc(ballot & ((1u << lane) - 1u));
        if (lane == 0) warpCount[warp] = __popc(ballot);
        __syncthreads();
        int warpOffset = 0, chunkTotal = 0;
        for (int w = 0; w < DISPATCH_THREADS / 32; w++)
        {
            if (w < warp) warpOffset += warpCount[w];
            chunkTotal += warpCount[w];
        }
        if (flag)
        {
            const int slot = base + running + warpOffset + rank;
            permutedToken[slot] = p / k;
            pairSlot[p] = slot;
        }
        running += chunkTotal;
        __syncthreads();
    }
}

extern "C" __global__ void moe_combine_f32(
    float* __restrict__ output,
    const float* __restrict__ expertOut,
    const int* __restrict__ pairSlot,
    const float* __restrict__ topkWeight,
    int hidden,
    int k,
    int accumulate,
    int expertRows)
{
    const int t = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    float acc = 0.0f;
    for (int j = 0; j < k; j++)
    {
        const int slot = pairSlot[(long long)t * k + j];
        if (slot < 0 || slot >= expertRows) continue;
        acc = __fadd_rn(acc, __fmul_rn(topkWeight[(long long)t * k + j], expertOut[(long long)slot * hidden + c]));
    }
    float* o = output + (long long)t * hidden + c;
    *o = accumulate ? __fadd_rn(*o, acc) : acc;
}
