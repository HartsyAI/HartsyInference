namespace HartsyInference.Cuda;

// General Sol-Attn in F32 (Kernels/attention/sol_attn_f32.cu): the five stages of SolAttentionReference.
public sealed partial class CudaKernels
{
    private CudaModule? _solAttentionModule;
    private nint _solSummaries, _solKeyStats, _solPooled, _solRoute, _solAttend, _solCoarseLogits, _solCoarse;

    /// <summary>True when sol_attn_f32.ptx loaded. A build without the file leaves this false, and Sol-Attn is then refused.</summary>
    public bool HasSolAttention => _solSummaries != 0 && _solKeyStats != 0 && _solPooled != 0 && _solRoute != 0 && _solAttend != 0
        && _solCoarseLogits != 0 && _solCoarse != 0;

    // Optional module: absence leaves the primitive unsupported on this backend instead of failing construction.
    private void LoadSolAttentionKernels()
    {
        string path = Ptx("sol_attn_f32");
        if (!CanLoad(path)) return;
        _solAttentionModule = LoadOwnedModule(path, required: false);
        _solSummaries = _solAttentionModule.GetFunction("sol_summaries");
        _solKeyStats = _solAttentionModule.GetFunction("sol_key_stats");
        _solPooled = _solAttentionModule.GetFunction("sol_pooled");
        _solRoute = _solAttentionModule.GetFunction("sol_route");
        _solAttend = _solAttentionModule.GetFunction("sol_attend");
        _solCoarseLogits = _solAttentionModule.GetFunction("sol_coarse_logits");
        _solCoarse = _solAttentionModule.GetFunction("sol_coarse");
    }

    /// <summary>The coarse gate's first stage: the natural-unit score of every (query block, key block), <c>[B, H, blocks, blocks]</c>.</summary>
    public unsafe void LaunchSolCoarseLogits(ulong logits, ulong centroids, ulong keyMeans, int batch, int heads, int dim, int blocks, float scale,
        nint stream)
    {
        ulong lA = logits, cA = centroids, kmA = keyMeans;
        int hA = heads, dA = dim, bA = blocks;
        float sA = scale;
        void** a = stackalloc void*[7];
        a[0] = &lA; a[1] = &cA; a[2] = &kmA; a[3] = &hA; a[4] = &dA; a[5] = &bA; a[6] = &sA;
        CudaDriverApi.cuLaunchKernel(_solCoarseLogits, (uint)blocks, (uint)heads, (uint)batch, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>The coarse gate's second stage: each query block's softmax over blocks mixing the blocks' mean values,
    /// <c>[B, H, blocks, D]</c>.</summary>
    public unsafe void LaunchSolCoarse(ulong coarse, ulong logits, ulong valueSums, ulong lengths, int batch, int heads, int dim, int blocks,
        nint stream)
    {
        ulong cA = coarse, lA = logits, vsA = valueSums, lenA = lengths;
        int hA = heads, dA = dim, bA = blocks;
        void** a = stackalloc void*[7];
        a[0] = &cA; a[1] = &lA; a[2] = &vsA; a[3] = &lenA; a[4] = &hA; a[5] = &dA; a[6] = &bA;
        CudaDriverApi.cuLaunchKernel(_solCoarse, (uint)blocks, (uint)heads, (uint)batch, 128, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Stage 1: mean K, summed V and mean Q of each block's live rows, each <c>[B, H, blocks, D]</c>.</summary>
    public unsafe void LaunchSolSummaries(ulong keyMeans, ulong valueSums, ulong centroids, ulong q, ulong k, ulong v, ulong lengths,
        int batch, int tokens, int heads, int dim, int blocks, nint stream)
    {
        ulong kmA = keyMeans, vsA = valueSums, cA = centroids, qA = q, kA = k, vA = v, lenA = lengths;
        int tA = tokens, hA = heads, dA = dim, bA = blocks;
        void** a = stackalloc void*[11];
        a[0] = &kmA; a[1] = &vsA; a[2] = &cA; a[3] = &qA; a[4] = &kA; a[5] = &vA; a[6] = &lenA; a[7] = &tA; a[8] = &hA; a[9] = &dA; a[10] = &bA;
        CudaDriverApi.cuLaunchKernel(_solSummaries, (uint)blocks, (uint)heads, (uint)batch, 128, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Stage 2: the mean of the block means and their variance, each <c>[B, H, D]</c>.</summary>
    public unsafe void LaunchSolKeyStats(ulong keyMean, ulong variance, ulong keyMeans, int batch, int heads, int dim, int blocks, nint stream)
    {
        ulong mA = keyMean, varA = variance, kmA = keyMeans;
        int hA = heads, dA = dim, bA = blocks;
        void** a = stackalloc void*[6];
        a[0] = &mA; a[1] = &varA; a[2] = &kmA; a[3] = &hA; a[4] = &dA; a[5] = &bA;
        CudaDriverApi.cuLaunchKernel(_solKeyStats, (uint)heads, (uint)batch, 1, 128, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Stage 3: the pooled score of every (query block, key block), <c>[B, H, blocks, blocks]</c>, in base-2 units.</summary>
    public unsafe void LaunchSolPooled(ulong pooled, ulong centroids, ulong keyMeans, ulong keyMean, int batch, int heads, int dim, int blocks,
        float log2Scale, nint stream)
    {
        ulong pA = pooled, cA = centroids, kmA = keyMeans, mA = keyMean;
        int hA = heads, dA = dim, bA = blocks;
        float sA = log2Scale;
        void** a = stackalloc void*[8];
        a[0] = &pA; a[1] = &cA; a[2] = &kmA; a[3] = &mA; a[4] = &hA; a[5] = &dA; a[6] = &bA; a[7] = &sA;
        CudaDriverApi.cuLaunchKernel(_solPooled, (uint)blocks, (uint)heads, (uint)batch, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Stage 4: which key blocks each query block reads token by token, one byte each, <c>[B, H, blocks, blocks]</c>.</summary>
    /// <param name="budget">Blocks routed by rank; below zero routes by the <paramref name="tau"/> threshold.</param>
    public unsafe void LaunchSolRoute(ulong reads, ulong pooled, ulong centroids, ulong variance, int batch, int heads, int dim, int blocks,
        float log2Scale, float tau, int budget, int sinkStart, int sinkEnd, int sinkQueryStart, int sinkQueryEnd, nint stream)
    {
        ulong rA = reads, pA = pooled, cA = centroids, varA = variance;
        int hA = heads, dA = dim, bA = blocks, budgetA = budget, ssA = sinkStart, seA = sinkEnd, sqsA = sinkQueryStart, sqeA = sinkQueryEnd;
        float sA = log2Scale, tauA = tau;
        void** a = stackalloc void*[14];
        a[0] = &rA; a[1] = &pA; a[2] = &cA; a[3] = &varA; a[4] = &hA; a[5] = &dA; a[6] = &bA; a[7] = &sA; a[8] = &tauA; a[9] = &budgetA;
        a[10] = &ssA; a[11] = &seA; a[12] = &sqsA; a[13] = &sqeA;
        CudaDriverApi.cuLaunchKernel(_solRoute, (uint)blocks, (uint)heads, (uint)batch, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Stage 5: one softmax per query token over the exactly read keys and, with the tail, the pooled blocks.</summary>
    /// <param name="bias">Device pointer to a per-key bias in natural-log units, or 0.</param>
    /// <param name="biasStride">0 for a bias of <c>[T]</c>, the token count for one of <c>[B, T]</c>.</param>
    /// <param name="gate">Device pointer to the coarse gate <c>[B, T, H, D]</c>, or 0 for none.</param>
    /// <param name="coarse">Device pointer to <see cref="LaunchSolCoarse"/>'s output; read only with a gate.</param>
    public unsafe void LaunchSolAttend(ulong output, ulong q, ulong k, ulong v, ulong keyMean, ulong pooled, ulong valueSums, ulong reads,
        ulong lengths, ulong bias, int biasStride, int batch, int tokens, int heads, int dim, int blocks, float log2Scale, bool tail,
        ulong gate, ulong coarse, nint stream)
    {
        ulong oA = output, qA = q, kA = k, vA = v, mA = keyMean, pA = pooled, vsA = valueSums, rA = reads, lenA = lengths, biasA = bias;
        ulong gateA = gate, coarseA = coarse;
        int strideA = biasStride, tA = tokens, hA = heads, dA = dim, bA = blocks, tailA = tail ? 1 : 0;
        float sA = log2Scale;
        void** a = stackalloc void*[19];
        a[0] = &oA; a[1] = &qA; a[2] = &kA; a[3] = &vA; a[4] = &mA; a[5] = &pA; a[6] = &vsA; a[7] = &rA; a[8] = &lenA; a[9] = &biasA;
        a[10] = &strideA; a[11] = &tA; a[12] = &hA; a[13] = &dA; a[14] = &bA; a[15] = &sA; a[16] = &tailA; a[17] = &gateA; a[18] = &coarseA;
        CudaDriverApi.cuLaunchKernel(_solAttend, (uint)tokens, (uint)heads, (uint)batch, 64, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
