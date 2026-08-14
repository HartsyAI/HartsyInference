// Kernels of the LTX-2.5 NADiffusionDecoder's neighborhood-attention block: the 3-axis interleaved RoPE and
// the 3D neighborhood attention itself. Both mirror the managed reference in IBackend/
// LtxVideo25NeighborhoodAttention3d exactly — including NATTEN's slide-inward window rule — so the CUDA path
// can be diffed against it. F32 only; every tensor in this decoder is F32.

// Start of the attended window along one axis. The window is always exactly `kernel` wide and slides inward
// at the borders rather than truncating, so every query attends the same number of keys. `kernel` is
// pre-clamped to `length` by the caller.
__device__ __forceinline__ int na3d_window_start(int index, int length, int kernel)
{
    int start = index - kernel / 2;
    int last = length - kernel;
    if (start < 0) start = 0;
    if (start > last) start = last;
    return start;
}

// 3D neighborhood attention over q/k/v[batch, T, H, W, heads, headDim] → out (same shape).
// One block per (batch, t, h, w, head) query; threads split the window for the score pass and the head_dim
// for the value pass, which keeps both passes coalesced. Dynamic shared memory holds the q row followed by
// the window's scores: (headDim + kt*kh*kw) floats.
extern "C" __global__ void ltx25_na3d_f32(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned int batch, unsigned int dimT, unsigned int dimH, unsigned int dimW,
    unsigned int heads, unsigned int headDim,
    unsigned int kt, unsigned int kh, unsigned int kw,
    float scale)
{
    extern __shared__ float smem[];
    float* sq = smem;                 // [headDim]  the query row
    float* sc = smem + headDim;       // [kt*kh*kw] scores, then softmax weights

    unsigned int window = kt * kh * kw;
    unsigned long long blocksPerHead = (unsigned long long)heads;
    unsigned long long gid = blockIdx.x;

    unsigned int head = (unsigned int)(gid % blocksPerHead);
    unsigned long long tok = gid / blocksPerHead;
    unsigned int iw = (unsigned int)(tok % dimW); tok /= dimW;
    unsigned int ih = (unsigned int)(tok % dimH); tok /= dimH;
    unsigned int it = (unsigned int)(tok % dimT); tok /= dimT;
    unsigned int b = (unsigned int)tok;
    if (b >= batch) return;

    unsigned long long headStride = headDim;
    unsigned long long wStride = (unsigned long long)heads * headDim;
    unsigned long long hStride = (unsigned long long)dimW * wStride;
    unsigned long long tStride = (unsigned long long)dimH * hStride;
    unsigned long long bStride = (unsigned long long)dimT * tStride;

    unsigned long long qOff = (unsigned long long)b * bStride + (unsigned long long)it * tStride
        + (unsigned long long)ih * hStride + (unsigned long long)iw * wStride
        + (unsigned long long)head * headStride;

    unsigned int tid = threadIdx.x, nthreads = blockDim.x;
    for (unsigned int d = tid; d < headDim; d += nthreads) sq[d] = q[qOff + d];
    __syncthreads();

    int t0 = na3d_window_start((int)it, (int)dimT, (int)kt);
    int h0 = na3d_window_start((int)ih, (int)dimH, (int)kh);
    int w0 = na3d_window_start((int)iw, (int)dimW, (int)kw);
    unsigned long long winBase = (unsigned long long)b * bStride + (unsigned long long)t0 * tStride
        + (unsigned long long)h0 * hStride + (unsigned long long)w0 * wStride
        + (unsigned long long)head * headStride;

    // Scores: one thread per window position, each doing a serial headDim-long dot against the shared q row.
    float localMax = -3.402823466e+38f;   // nvrtc compiles this TU without headers, so no INFINITY macro
    for (unsigned int n = tid; n < window; n += nthreads)
    {
        unsigned int ww = n % kw, rest = n / kw;
        unsigned int wh = rest % kh, wt = rest / kh;
        unsigned long long kOff = winBase + (unsigned long long)wt * tStride
            + (unsigned long long)wh * hStride + (unsigned long long)ww * wStride;
        float dot = 0.0f;
        for (unsigned int d = 0; d < headDim; d++) dot += sq[d] * k[kOff + d];
        dot *= scale;
        sc[n] = dot;
        if (dot > localMax) localMax = dot;
    }

    // Block-reduce the max, then the softmax denominator, through a single warp-partials buffer.
    __shared__ float partials[32];
    unsigned int lane = tid & 31u, warp = tid >> 5, warps = (nthreads + 31u) >> 5;
    for (int off = 16; off > 0; off >>= 1)
    {
        float other = __shfl_down_sync(0xffffffffu, localMax, off);
        if (other > localMax) localMax = other;
    }
    if (lane == 0) partials[warp] = localMax;
    __syncthreads();
    if (tid == 0)
    {
        float m = partials[0];
        for (unsigned int i = 1; i < warps; i++) if (partials[i] > m) m = partials[i];
        partials[0] = m;
    }
    __syncthreads();
    float maxScore = partials[0];
    __syncthreads();

    float localSum = 0.0f;
    for (unsigned int n = tid; n < window; n += nthreads)
    {
        float e = expf(sc[n] - maxScore);
        sc[n] = e;
        localSum += e;
    }
    for (int off = 16; off > 0; off >>= 1) localSum += __shfl_down_sync(0xffffffffu, localSum, off);
    if (lane == 0) partials[warp] = localSum;
    __syncthreads();
    if (tid == 0)
    {
        float s = 0.0f;
        for (unsigned int i = 0; i < warps; i++) s += partials[i];
        partials[0] = s;
    }
    __syncthreads();
    float sum = partials[0];
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    // Values: one thread per head_dim lane, walking the window with running offsets so the v reads stay
    // contiguous across the block.
    for (unsigned int d = tid; d < headDim; d += nthreads)
    {
        float acc = 0.0f;
        unsigned int n = 0;
        for (unsigned int wt = 0; wt < kt; wt++)
        {
            unsigned long long tBase = winBase + (unsigned long long)wt * tStride;
            for (unsigned int wh = 0; wh < kh; wh++)
            {
                unsigned long long hBase = tBase + (unsigned long long)wh * hStride;
                for (unsigned int ww = 0; ww < kw; ww++)
                    acc += sc[n++] * v[hBase + (unsigned long long)ww * wStride + d];
            }
        }
        out[qOff + d] = acc * inv;
    }
}

// 3-axis interleaved RoPE on x[1, t, h, w, heads, headDim], in place. Each head's channels split into three
// contiguous chunks — T at offset 0, H at offset splitT, W at offset splitT+splitH — rotated by the token's
// t / h / w coordinate respectively, over interleaved (2i, 2i+1) pairs. The cos/sin tables are built on the
// host so the F32 angle rounding matches the reference exactly. One thread per (token, head, pair).
extern "C" __global__ void ltx25_na_rope3d_f32(
    float* __restrict__ x,
    const float* __restrict__ cosT, const float* __restrict__ sinT,
    const float* __restrict__ cosH, const float* __restrict__ sinH,
    const float* __restrict__ cosW, const float* __restrict__ sinW,
    unsigned int dimT, unsigned int dimH, unsigned int dimW,
    unsigned int heads, unsigned int headDim,
    unsigned int pairsT, unsigned int pairsH, unsigned int pairsW,
    unsigned int offsetH, unsigned int offsetW)
{
    unsigned int pairsTotal = pairsT + pairsH + pairsW;
    unsigned long long total = (unsigned long long)dimT * dimH * dimW * heads * pairsTotal;
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    unsigned int p = (unsigned int)(gid % pairsTotal);
    unsigned long long rest = gid / pairsTotal;
    unsigned int head = (unsigned int)(rest % heads); rest /= heads;
    unsigned int wi = (unsigned int)(rest % dimW); rest /= dimW;
    unsigned int hi = (unsigned int)(rest % dimH); rest /= dimH;
    unsigned int ti = (unsigned int)rest;

    const float* cosTab; const float* sinTab;
    unsigned int chunkOffset, pairIndex, tableRow, pairs;
    if (p < pairsT)
    {
        cosTab = cosT; sinTab = sinT; chunkOffset = 0u; pairIndex = p; tableRow = ti; pairs = pairsT;
    }
    else if (p < pairsT + pairsH)
    {
        cosTab = cosH; sinTab = sinH; chunkOffset = offsetH; pairIndex = p - pairsT; tableRow = hi; pairs = pairsH;
    }
    else
    {
        cosTab = cosW; sinTab = sinW; chunkOffset = offsetW; pairIndex = p - pairsT - pairsH; tableRow = wi; pairs = pairsW;
    }

    unsigned long long tokenBase = (((unsigned long long)ti * dimH + hi) * dimW + wi)
        * (unsigned long long)heads * headDim;
    unsigned long long rowOff = tokenBase + (unsigned long long)head * headDim + chunkOffset;
    unsigned long long tabOff = (unsigned long long)tableRow * pairs + pairIndex;

    float even = x[rowOff + 2u * pairIndex], odd = x[rowOff + 2u * pairIndex + 1u];
    float c = cosTab[tabOff], s = sinTab[tabOff];
    x[rowOff + 2u * pairIndex] = even * c - odd * s;
    x[rowOff + 2u * pairIndex + 1u] = even * s + odd * c;
}
