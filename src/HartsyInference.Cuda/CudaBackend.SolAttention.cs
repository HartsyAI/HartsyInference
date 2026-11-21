using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// General Sol-Attn on the device (Kernels/attention/sol_attn_f32.cu).
public sealed partial class CudaBackend
{
    /// <summary>The most tokens a head dimension the kernels take: the token pass stages a query row of up to 256 values.</summary>
    private const int SolAttentionMaxHeadDim = 256;

    /// <inheritdoc/>
    public bool HasSolAttention
    {
        get
        {
            EnsureKernels();
            return _kernels is { HasSolAttention: true };
        }
    }

    /// <inheritdoc/>
    /// <remarks>F32, in the reference's five stages. The routing stages compute the reference's scores bit for bit, so the
    /// routing is the reference's; the token pass is ordinary F32 arithmetic. The coarse gate is not implemented: a call
    /// that asks for one is refused. Asking for the routing report reads it back from the device, which waits for the
    /// stream.</remarks>
    public unsafe void SolAttention(Tensor output, Tensor query, Tensor key, Tensor value, SolAttentionOptions options, Span<bool> routed = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        SolAttentionReference.Check(output, query, key, value, options);
        int batch = (int)query.Shape[0], tokens = (int)query.Shape[1], heads = (int)query.Shape[2], dim = (int)query.Shape[3];
        int blocks = SolAttentionReference.Blocks(tokens);
        long pairs = (long)batch * heads * blocks * blocks;
        if (!routed.IsEmpty && routed.Length != pairs)
            throw new ArgumentException($"The routing report holds {routed.Length} entries; [B, H, blocks, blocks] needs {pairs}.", nameof(routed));
        using OpScope _op = EnterOp(nameof(SolAttention));
        EnsureKernels();
        if (_kernels is not { HasSolAttention: true })
            throw new NotSupportedException("SolAttention is not available: sol_attn_f32.ptx is not in the Ptx folder.");
        if (options.CoarseGate is not null) throw new NotSupportedException("CUDA Sol-Attn has no coarse gate; run SolAttentionReference for it.");
        if (dim > SolAttentionMaxHeadDim || !CudaKernels.GridYzFits(heads, batch) || pairs > int.MaxValue)
            throw new NotSupportedException($"CUDA Sol-Attn takes a head dim up to {SolAttentionMaxHeadDim} and a grid of heads x batch; got {query.Shape}.");

        float scale = options.Scale ?? 1f / MathF.Sqrt(dim);
        float log2Scale = scale * SolAttentionReference.Log2E;
        int sinkStart = Math.Min(options.SinkBlocks.Start, blocks), sinkEnd = Math.Min(options.SinkBlocks.End, blocks);
        int sinkQueryStart = Math.Min(options.SinkQueries.Start, blocks), sinkQueryEnd = Math.Min(options.SinkQueries.End, blocks);
        int budget = options.TopKRatio > 0 ? SolAttentionReference.TopKBudget(options.TopKRatio, blocks, Math.Max(0, sinkEnd - sinkStart)) : -1;
        Tensor? bias = options.KeyBias;
        int biasStride = bias is { } b && b.Shape.Rank == 2 && b.Shape[0] == batch && batch > 1 ? tokens : 0;

        using Tensor lengths = new(new TensorShape(blocks), DType.I32);
        SolAttentionReference.LiveLengths(options.BlockLengths, tokens, blocks).CopyTo(lengths.AsSpan<int>());

        nuint summaryBytes = (nuint)((long)batch * heads * blocks * dim * sizeof(float)), statBytes = (nuint)((long)batch * heads * dim * sizeof(float));
        nuint pooledBytes = (nuint)(pairs * sizeof(float)), readBytes = (nuint)pairs, outBytes = GpuTransferHelper.ByteSize(output);
        ulong pQ = 0, pK = 0, pV = 0, pLengths = 0, pBias = 0, pOut = 0;
        ulong pKeyMeans = 0, pValueSums = 0, pCentroids = 0, pKeyMean = 0, pVariance = 0, pPooled = 0, pReads = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            pLengths = GpuTransferHelper.CopyToDevice(lengths);
            AttentionEvidence.Observed(AttentionEvidenceKind.StagingCopy, "cuda.sol.lengths", blocks * sizeof(int));
            if (bias is not null) pBias = GpuTransferHelper.CopyToDevice(bias);
            pKeyMeans = GpuTransferHelper.AllocateDevice(summaryBytes);
            pValueSums = GpuTransferHelper.AllocateDevice(summaryBytes);
            pCentroids = GpuTransferHelper.AllocateDevice(summaryBytes);
            pKeyMean = GpuTransferHelper.AllocateDevice(statBytes);
            pVariance = GpuTransferHelper.AllocateDevice(statBytes);
            pPooled = GpuTransferHelper.AllocateDevice(pooledBytes);
            pReads = GpuTransferHelper.AllocateDevice(readBytes);
            AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, "cuda.sol.workspace",
                3 * (long)summaryBytes + 2 * (long)statBytes + (long)pooledBytes + (long)readBytes, count: 7);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            nint stream = _stream.Handle;
            _kernels.LaunchSolSummaries(pKeyMeans, pValueSums, pCentroids, pQ, pK, pV, pLengths, batch, tokens, heads, dim, blocks, stream);
            _kernels.LaunchSolKeyStats(pKeyMean, pVariance, pKeyMeans, batch, heads, dim, blocks, stream);
            _kernels.LaunchSolPooled(pPooled, pCentroids, pKeyMeans, pKeyMean, batch, heads, dim, blocks, log2Scale, stream);
            _kernels.LaunchSolRoute(pReads, pPooled, pCentroids, pVariance, batch, heads, dim, blocks, log2Scale, options.Tau, budget,
                sinkStart, sinkEnd, sinkQueryStart, sinkQueryEnd, stream);
            _kernels.LaunchSolAttend(pOut, pQ, pK, pV, pKeyMean, pPooled, pValueSums, pReads, pLengths, pBias, biasStride, batch, tokens, heads, dim,
                blocks, log2Scale, options.Tail, stream);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;

            if (!routed.IsEmpty)
            {
                _stream.Synchronize();
                byte[] report = new byte[pairs];
                fixed (byte* p = report) CudaMemory.CopyDeviceToHost(p, pReads, readBytes);
                for (int i = 0; i < report.Length; i++) routed[i] = report[i] != 0;
            }
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            foreach (ulong pointer in (ReadOnlySpan<ulong>)[pQ, pK, pV, pLengths, pBias, pKeyMeans, pValueSums, pCentroids, pKeyMean, pVariance, pPooled, pReads])
                GpuTransferHelper.FreeDevice(pointer);
        }
    }
}
