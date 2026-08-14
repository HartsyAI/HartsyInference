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

// ---------------------------------------------------------------------------------------------------------
// Query-tiled variant of the same attention. The naive kernel above re-reads a whole kt*kh*kw window per
// (query, head), which at the stage-5 trunk (window 11^3) is ~1300x more traffic than the data is worth.
// Adjacent queries share almost all of that window, so a block here owns a TH x TW tile of queries at ONE
// t and one head, and streams the tile's *union* window through shared memory once. Traffic drops by
// TH*TW*window / union; compute rises by union/window, which is the trade this kernel is making.
//
// The tile is deliberately 1 deep in T. Temporal chunking re-decodes the volume in frame windows, and the
// chunked/un-chunked decode must stay bit-identical: a tile spanning T would put the same global query into
// differently-aligned tiles in the two arms, changing its accumulation order. H and W are never chunked, so
// tiling them is safe. Within a tile every query walks its own window in the same (t, h, w) ascending order
// no matter where the tile falls, which is what keeps the two arms identical.
//
// Softmax is online (flash-style): the tile's union does not fit as a score matrix, so the running max and
// denominator are carried across chunks and the accumulator is rescaled when the max moves. Positions of the
// union that fall outside a given query's own window are masked to -FLT_MAX and contribute exp() == 0, which
// is an exact no-op on both the denominator and the accumulator.
//
// head_dim is fixed at 64 (the only value the shipped decoder uses) so the register tile is compile-time.

#define NA3D_HD 64
#define NA3D_CHUNK 32
#define NA3D_THREADS 256
#define NA3D_NEG_INF (-3.402823466e+38f)

template<int TH, int TW>
__device__ __forceinline__ void ltx25_na3d_tiled_impl(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned int batch, unsigned int dimT, unsigned int dimH, unsigned int dimW,
    unsigned int heads,
    unsigned int kt, unsigned int kh, unsigned int kw,
    float scale)
{
    const int HD = NA3D_HD;
    const int CHUNK = NA3D_CHUNK;
    const int NT = NA3D_THREADS;
    const int TQ = TH * TW;
    const int QPT = 4;                 // queries each thread accumulates
    const int QG = TQ / QPT;           // query groups per block
    const int LANES = NT / QG;         // threads cooperating on one query group
    const int PPT = CHUNK / LANES;     // window positions each thread scores
    const int DPT = HD / LANES;        // head_dim lanes each thread accumulates

    __shared__ float qs[TQ][HD + 1];
    __shared__ float ks[CHUNK][HD + 1];
    __shared__ float vs[CHUNK][HD + 1];
    __shared__ float sc[TQ][CHUNK + 1];
    __shared__ int qHs[TQ], qWs[TQ];
    __shared__ unsigned char qOk[TQ];
    __shared__ int pT[CHUNK], pH[CHUNK], pW[CHUNK];

    unsigned int hTiles = (dimH + TH - 1) / TH;
    unsigned int wTiles = (dimW + TW - 1) / TW;
    unsigned long long gid = blockIdx.x;
    unsigned int head = (unsigned int)(gid % heads); gid /= heads;
    unsigned int wTile = (unsigned int)(gid % wTiles); gid /= wTiles;
    unsigned int hTile = (unsigned int)(gid % hTiles); gid /= hTiles;
    unsigned int it = (unsigned int)(gid % dimT); gid /= dimT;
    unsigned int b = (unsigned int)gid;
    if (b >= batch) return;

    unsigned long long wStride = (unsigned long long)heads * HD;
    unsigned long long hStride = (unsigned long long)dimW * wStride;
    unsigned long long tStride = (unsigned long long)dimH * hStride;
    unsigned long long bStride = (unsigned long long)dimT * tStride;
    unsigned long long baseOff = (unsigned long long)b * bStride + (unsigned long long)head * HD;

    int h0 = (int)hTile * TH, w0 = (int)wTile * TW;
    int tStart = na3d_window_start((int)it, (int)dimT, (int)kt);
    int tid = (int)threadIdx.x;

    for (int qi = tid; qi < TQ; qi += NT)
    {
        int hh = h0 + qi / TW, ww = w0 + qi % TW;
        bool ok = hh < (int)dimH && ww < (int)dimW;
        qOk[qi] = ok ? 1 : 0;
        qHs[qi] = ok ? na3d_window_start(hh, (int)dimH, (int)kh) : 0;
        qWs[qi] = ok ? na3d_window_start(ww, (int)dimW, (int)kw) : 0;
    }
    for (int idx = tid; idx < TQ * HD; idx += NT)
    {
        int qi = idx / HD, d = idx - qi * HD;
        int hh = h0 + qi / TW, ww = w0 + qi % TW;
        float val = 0.0f;
        if (hh < (int)dimH && ww < (int)dimW)
        {
            val = q[baseOff + (unsigned long long)it * tStride
                + (unsigned long long)hh * hStride + (unsigned long long)ww * wStride + d];
        }
        qs[qi][d] = val;
    }

    // Union of the tile's windows. Window starts are monotonic in the query index, so the first and last
    // in-range query of the tile bracket every window the tile can produce.
    int hLast = h0 + TH - 1; if (hLast > (int)dimH - 1) hLast = (int)dimH - 1;
    int wLast = w0 + TW - 1; if (wLast > (int)dimW - 1) wLast = (int)dimW - 1;
    int uh0 = na3d_window_start(h0, (int)dimH, (int)kh);
    int uw0 = na3d_window_start(w0, (int)dimW, (int)kw);
    int uhN = na3d_window_start(hLast, (int)dimH, (int)kh) + (int)kh - uh0;
    int uwN = na3d_window_start(wLast, (int)dimW, (int)kw) + (int)kw - uw0;
    int total = (int)kt * uhN * uwN;

    int qg = tid / LANES, lane = tid % LANES;
    int qBase = qg * QPT, pBase = lane * PPT, dBase = lane * DPT;

    // Running softmax state lives in registers, replicated across the LANES threads that share a query: every
    // update below is a shuffle reduction over exactly those lanes, so the copies cannot drift, and the
    // alternative — one thread per query scanning the chunk in shared memory — idles most of the block and costs
    // two extra barriers per chunk.
    float acc[QPT][DPT], runMax[QPT], runSum[QPT];
#pragma unroll
    for (int i = 0; i < QPT; i++)
    {
        runMax[i] = NA3D_NEG_INF;
        runSum[i] = 0.0f;
#pragma unroll
        for (int jj = 0; jj < DPT; jj++) acc[i][jj] = 0.0f;
    }

    for (int chunkStart = 0; chunkStart < total; chunkStart += CHUNK)
    {
        for (int p = tid; p < CHUNK; p += NT)
        {
            int lin = chunkStart + p;
            if (lin >= total) { pH[p] = -1; pW[p] = 0; pT[p] = 0; continue; }
            int uw = lin % uwN, rest = lin / uwN;
            pT[p] = tStart + rest / uhN;
            pH[p] = uh0 + rest % uhN;
            pW[p] = uw0 + uw;
        }
        __syncthreads();

        for (int idx = tid; idx < CHUNK * HD; idx += NT)
        {
            int p = idx / HD, d = idx - p * HD;
            float kv = 0.0f, vv = 0.0f;
            if (pH[p] >= 0)
            {
                unsigned long long off = baseOff + (unsigned long long)pT[p] * tStride
                    + (unsigned long long)pH[p] * hStride + (unsigned long long)pW[p] * wStride + d;
                kv = k[off]; vv = v[off];
            }
            ks[p][d] = kv; vs[p][d] = vv;
        }
        __syncthreads();

        float sAcc[QPT][PPT];
#pragma unroll
        for (int i = 0; i < QPT; i++)
#pragma unroll
            for (int j = 0; j < PPT; j++) sAcc[i][j] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < HD; d++)
        {
            float qv[QPT], kv[PPT];
#pragma unroll
            for (int i = 0; i < QPT; i++) qv[i] = qs[qBase + i][d];
#pragma unroll
            for (int j = 0; j < PPT; j++) kv[j] = ks[pBase + j][d];
#pragma unroll
            for (int i = 0; i < QPT; i++)
#pragma unroll
                for (int j = 0; j < PPT; j++) sAcc[i][j] += qv[i] * kv[j];
        }
#pragma unroll
        for (int i = 0; i < QPT; i++)
        {
            int qi = qBase + i;
            float mx = NA3D_NEG_INF;
#pragma unroll
            for (int j = 0; j < PPT; j++)
            {
                int p = pBase + j;
                float val = NA3D_NEG_INF;
                if (qOk[qi] && pH[p] >= 0)
                {
                    int dh = pH[p] - qHs[qi], dw = pW[p] - qWs[qi];
                    if (dh >= 0 && dh < (int)kh && dw >= 0 && dw < (int)kw) val = sAcc[i][j] * scale;
                }
                sAcc[i][j] = val;
                if (val > mx) mx = val;
            }
            for (int off = LANES >> 1; off > 0; off >>= 1)
            {
                float other = __shfl_xor_sync(0xffffffffu, mx, off);
                if (other > mx) mx = other;
            }
            float newMax = runMax[i] > mx ? runMax[i] : mx;
            // Nothing valid has been seen yet — exp(-FLT_MAX + FLT_MAX) would be 1, not 0, so skip the chunk.
            bool dead = newMax == NA3D_NEG_INF;
            float rescale = dead ? 1.0f : expf(runMax[i] - newMax);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PPT; j++)
            {
                float e = dead ? 0.0f : expf(sAcc[i][j] - newMax);
                sc[qi][pBase + j] = e;
                sum += e;
            }
            for (int off = LANES >> 1; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            runMax[i] = newMax;
            runSum[i] = runSum[i] * rescale + sum;
#pragma unroll
            for (int jj = 0; jj < DPT; jj++) acc[i][jj] *= rescale;
        }
        __syncthreads();

        for (int p = 0; p < CHUNK; p++)
        {
            float pv[QPT], vv[DPT];
#pragma unroll
            for (int i = 0; i < QPT; i++) pv[i] = sc[qBase + i][p];
#pragma unroll
            for (int jj = 0; jj < DPT; jj++) vv[jj] = vs[p][dBase + jj];
#pragma unroll
            for (int i = 0; i < QPT; i++)
#pragma unroll
                for (int jj = 0; jj < DPT; jj++) acc[i][jj] += pv[i] * vv[jj];
        }
    }

#pragma unroll
    for (int i = 0; i < QPT; i++)
    {
        int qi = qBase + i;
        if (!qOk[qi]) continue;
        int hh = h0 + qi / TW, ww = w0 + qi % TW;
        unsigned long long off = baseOff + (unsigned long long)it * tStride
            + (unsigned long long)hh * hStride + (unsigned long long)ww * wStride;
        float inv = runSum[i] > 0.0f ? 1.0f / runSum[i] : 0.0f;
#pragma unroll
        for (int jj = 0; jj < DPT; jj++) out[off + dBase + jj] = acc[i][jj] * inv;
    }
}

extern "C" __global__ __launch_bounds__(NA3D_THREADS) void ltx25_na3d_tiled8x8_f32(
    float* __restrict__ out,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    unsigned int batch, unsigned int dimT, unsigned int dimH, unsigned int dimW,
    unsigned int heads, unsigned int headDim,
    unsigned int kt, unsigned int kh, unsigned int kw,
    float scale)
{
    ltx25_na3d_tiled_impl<8, 8>(out, q, k, v, batch, dimT, dimH, dimW, heads, kt, kh, kw, scale);
}

extern "C" __global__ __launch_bounds__(NA3D_THREADS) void ltx25_na3d_tiled4x8_f32(
    float* __restrict__ out,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    unsigned int batch, unsigned int dimT, unsigned int dimH, unsigned int dimW,
    unsigned int heads, unsigned int headDim,
    unsigned int kt, unsigned int kh, unsigned int kw,
    float scale)
{
    ltx25_na3d_tiled_impl<4, 8>(out, q, k, v, batch, dimT, dimH, dimW, heads, kt, kh, kw, scale);
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

// Channels-last 3D pixel shuffle of one temporal chunk of the upsample projection. Each source token
// [row, projChannels] unpacks into a p1*p2*p3 sub-block of output tokens, taking channel `packed` of the
// source to channel `c` of the destination, where packed = ((c*p1 + i1)*p2 + i2)*p3 + i3.
// One thread per SOURCE element, so reads are fully coalesced and the strided access is on the write side.
// `dropped` shifts the temporal axis by one frame (the causal shuffle's duplicated leading frame); the
// source elements that map before frame 0 have no destination and simply return.
extern "C" __global__ void ltx25_pixel_shuffle_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int start, int h, int w, int p1, int p2, int p3,
    int outChannels, int dropped, int outH, int outW,
    unsigned long long total)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int projChannels = p1 * p2 * p3 * outChannels;
    unsigned long long srcRow = i / (unsigned long long)projChannels;
    int packed = (int)(i - srcRow * (unsigned long long)projChannels);

    int i3 = packed % p3; int rest = packed / p3;
    int i2 = rest % p2; rest /= p2;
    int i1 = rest % p1; int c = rest / p1;

    int wi = (int)(srcRow % (unsigned long long)w);
    unsigned long long r = srcRow / (unsigned long long)w;
    int hi = (int)(r % (unsigned long long)h);
    int tLocal = (int)(r / (unsigned long long)h);

    int frame = (tLocal + start) * p1 + i1 - dropped;
    if (frame < 0) return;

    long long dstRow = ((long long)frame * outH + (long long)hi * p2 + i2) * (long long)outW
                     + (long long)wi * p3 + i3;
    dst[dstRow * (long long)outChannels + c] = src[i];
}
