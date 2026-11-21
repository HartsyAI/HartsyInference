using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// General Sol-Attn on the device (Shaders/sol_attn_*.comp.glsl): the five stages of SolAttentionReference.
public sealed partial class VulkanBackend
{
    /// <summary>The widest head dimension the token pass stages in shared memory.</summary>
    private const int SolAttentionMaxHeadDim = 256;

    /// <inheritdoc/>
    public bool HasSolAttention => true;

    /// <inheritdoc/>
    /// <remarks>F32, in the reference's five stages. The routing stages are written to reproduce the reference's scores:
    /// their adds, subtracts and multiplies are exact and uncontracted where the driver keeps subnormal values, and a
    /// divide and a square root are exact only where the driver rounds them correctly; the Vulkan specification requires
    /// neither. The token pass and the coarse gate are ordinary F32 arithmetic. A
    /// call that asks for one is refused. Asking for the routing report reads it back, which waits for the device.</remarks>
    public void SolAttention(Tensor output, Tensor query, Tensor key, Tensor value, SolAttentionOptions options, Span<bool> routed = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        SolAttentionReference.Check(output, query, key, value, options);
        int batch = (int)query.Shape[0], tokens = (int)query.Shape[1], heads = (int)query.Shape[2], dim = (int)query.Shape[3];
        int blocks = SolAttentionReference.Blocks(tokens);
        long pairs = (long)batch * heads * blocks * blocks;
        if (!routed.IsEmpty && routed.Length != pairs)
            throw new ArgumentException($"The routing report holds {routed.Length} entries; [B, H, blocks, blocks] needs {pairs}.", nameof(routed));
        // The shaders index with 32-bit unsigned arithmetic, so the operands' element count must fit one.
        if (dim > SolAttentionMaxHeadDim || tokens > ushort.MaxValue || heads > ushort.MaxValue || batch > ushort.MaxValue || pairs > int.MaxValue
            || query.ElementCount > uint.MaxValue)
            throw new NotSupportedException(
                $"Vulkan Sol-Attn takes a head dim up to {SolAttentionMaxHeadDim}, up to {ushort.MaxValue} tokens, heads and sequences, and operands of under 2^32 elements; got {query.Shape}.");

        float scale = options.Scale ?? 1f / MathF.Sqrt(dim);
        float log2Scale = scale * SolAttentionReference.Log2E;
        int sinkStart = Math.Min(options.SinkBlocks.Start, blocks), sinkEnd = Math.Min(options.SinkBlocks.End, blocks);
        int sinkQueryStart = Math.Min(options.SinkQueries.Start, blocks), sinkQueryEnd = Math.Min(options.SinkQueries.End, blocks);
        int budget = options.TopKRatio > 0 ? SolAttentionReference.TopKBudget(options.TopKRatio, blocks, Math.Max(0, sinkEnd - sinkStart)) : -1;
        Tensor? bias = options.KeyBias;
        uint biasStride = bias is { } given && given.Shape.Rank == 2 && given.Shape[0] == batch && batch > 1 ? (uint)tokens : 0u;

        using OpScope _op = EnterOp();
        using Tensor lengths = new(new TensorShape(blocks), DType.I32);
        SolAttentionReference.LiveLengths(options.BlockLengths, tokens, blocks).CopyTo(lengths.AsSpan<int>());
        // The routing lives in a tensor so the report, when asked for, is read back the way any output is.
        using Tensor reads = new(new TensorShape(pairs), DType.I32);

        VulkanBuffer qBuf = GetBuffer(query), kBuf = GetBuffer(key), vBuf = GetBuffer(value), lenBuf = GetBuffer(lengths);
        AttentionEvidence.Observed(AttentionEvidenceKind.StagingCopy, "vulkan.sol.lengths", blocks * sizeof(int));
        // A buffer is always bound at the bias binding; without a bias the lengths stand in and are not read there.
        VulkanBuffer biasBuf = bias is null ? lenBuf : GetBuffer(bias);
        if (bias is not null)
            AttentionEvidence.Observed(AttentionEvidenceKind.StagingCopy, "vulkan.sol.bias", bias.ElementCount * sizeof(float));

        ulong floatBytes = sizeof(float);
        ulong summaryBytes = (ulong)batch * (ulong)heads * (ulong)blocks * (ulong)dim * floatBytes, statBytes = (ulong)batch * (ulong)heads * (ulong)dim * floatBytes;
        ulong pairBytes = (ulong)pairs * floatBytes;
        // Allocated inside the try, so one that fails leaves none of the earlier ones behind.
        VulkanBuffer? keyMeans = null, valueSums = null, centroids = null, keyMean = null, variance = null, pooled = null, readsBuf = null, outBuf = null;
        VulkanBuffer? coarseLogits = null, coarse = null;
        Tensor? gate = options.CoarseGate;
        // Buffers are always bound at the gate and coarse bindings; without a gate the lengths stand in and are not read.
        VulkanBuffer gateBuf = gate is null ? lenBuf : GetBuffer(gate);
        bool readsCached = false, outputCached = false;
        try
        {
            keyMeans = _xfer.AllocateDevice(summaryBytes);
            valueSums = _xfer.AllocateDevice(summaryBytes);
            centroids = _xfer.AllocateDevice(summaryBytes);
            keyMean = _xfer.AllocateDevice(statBytes);
            variance = _xfer.AllocateDevice(statBytes);
            pooled = _xfer.AllocateDevice(pairBytes);
            readsBuf = _xfer.AllocateDevice(pairBytes);
            outBuf = _xfer.AllocateDevice((ulong)output.ElementCount * floatBytes);
            if (gate is not null)
            {
                coarseLogits = _xfer.AllocateDevice(pairBytes);
                coarse = _xfer.AllocateDevice(summaryBytes);
                AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, "vulkan.sol.coarse", (long)(pairBytes + summaryBytes), count: 2);
            }
            AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, "vulkan.sol.workspace", (long)(3 * summaryBytes + 2 * statBytes + 2 * pairBytes), count: 7);
            Span<byte> pc = stackalloc byte[10 * 4];

            BinaryWriteUInt(pc, 0, (uint)tokens);
            BinaryWriteUInt(pc, 4, (uint)heads);
            BinaryWriteUInt(pc, 8, (uint)dim);
            BinaryWriteUInt(pc, 12, (uint)blocks);
            Span<ulong> summaryBufs = stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, lenBuf.Handle, keyMeans.Handle, valueSums.Handle, centroids.Handle };
            Dispatch(GetKernel("sol_attn_summaries", storageBufferCount: 7, default), summaryBufs, pc[..16], (uint)blocks, (uint)heads, (uint)batch);

            BinaryWriteUInt(pc, 0, (uint)heads);
            BinaryWriteUInt(pc, 4, (uint)dim);
            BinaryWriteUInt(pc, 8, (uint)blocks);
            Span<ulong> statBufs = stackalloc ulong[] { keyMeans.Handle, keyMean.Handle, variance.Handle };
            Dispatch(GetKernel("sol_attn_key_stats", storageBufferCount: 3, default), statBufs, pc[..12], (uint)heads, (uint)batch);

            BinaryWriteFloat(pc, 12, log2Scale);
            Span<ulong> pooledBufs = stackalloc ulong[] { centroids.Handle, keyMeans.Handle, keyMean.Handle, pooled.Handle };
            Dispatch(GetKernel("sol_attn_pooled", storageBufferCount: 4, default), pooledBufs, pc[..16], (uint)blocks, (uint)heads, (uint)batch);

            BinaryWriteFloat(pc, 16, options.Tau);
            BinaryWriteUInt(pc, 20, unchecked((uint)budget));
            BinaryWriteUInt(pc, 24, unchecked((uint)sinkStart));
            BinaryWriteUInt(pc, 28, unchecked((uint)sinkEnd));
            BinaryWriteUInt(pc, 32, unchecked((uint)sinkQueryStart));
            BinaryWriteUInt(pc, 36, unchecked((uint)sinkQueryEnd));
            Span<ulong> routeBufs = stackalloc ulong[] { pooled.Handle, centroids.Handle, variance.Handle, readsBuf.Handle };
            Dispatch(GetKernel("sol_attn_route", storageBufferCount: 4, default), routeBufs, pc, (uint)blocks, (uint)heads, (uint)batch);

            if (gate is not null)
            {
                BinaryWriteUInt(pc, 0, (uint)heads);
                BinaryWriteUInt(pc, 4, (uint)dim);
                BinaryWriteUInt(pc, 8, (uint)blocks);
                BinaryWriteFloat(pc, 12, scale);
                Span<ulong> logitBufs = stackalloc ulong[] { centroids.Handle, keyMeans.Handle, coarseLogits!.Handle };
                Dispatch(GetKernel("sol_attn_coarse_logits", storageBufferCount: 3, default), logitBufs, pc[..16], (uint)blocks, (uint)heads, (uint)batch);
                Span<ulong> coarseBufs = stackalloc ulong[] { coarseLogits.Handle, valueSums.Handle, lenBuf.Handle, coarse!.Handle };
                Dispatch(GetKernel("sol_attn_coarse", storageBufferCount: 4, default), coarseBufs, pc[..12], (uint)blocks, (uint)heads, (uint)batch);
            }

            BinaryWriteUInt(pc, 0, (uint)tokens);
            BinaryWriteUInt(pc, 4, (uint)heads);
            BinaryWriteUInt(pc, 8, (uint)dim);
            BinaryWriteUInt(pc, 12, (uint)blocks);
            BinaryWriteFloat(pc, 16, log2Scale);
            BinaryWriteUInt(pc, 20, options.Tail ? 1u : 0u);
            BinaryWriteUInt(pc, 24, biasStride);
            BinaryWriteUInt(pc, 28, bias is null ? 0u : 1u);
            BinaryWriteUInt(pc, 32, gate is null ? 0u : 1u);
            Span<ulong> attendBufs = stackalloc ulong[]
            {
                qBuf.Handle, kBuf.Handle, vBuf.Handle, keyMean.Handle, pooled.Handle, valueSums.Handle, readsBuf.Handle, lenBuf.Handle,
                biasBuf.Handle, outBuf.Handle, gateBuf.Handle, (coarse ?? lenBuf).Handle,
            };
            Dispatch(GetKernel("sol_attn_attend", storageBufferCount: 12, default), attendBufs, pc[..36], (uint)tokens, (uint)heads, (uint)batch);

            CacheOutput(output, outBuf);
            outputCached = true;
            CacheOutput(reads, readsBuf);
            readsCached = true;
        }
        catch (Exception)
        {
            if (!outputCached) outBuf?.Dispose();
            if (!readsCached) readsBuf?.Dispose();
            throw;
        }
        finally
        {
            foreach (VulkanBuffer? buffer in (ReadOnlySpan<VulkanBuffer?>)[keyMeans, valueSums, centroids, keyMean, variance, pooled, coarseLogits, coarse])
                if (buffer is not null) _xfer.FreeDevice(buffer);
        }
        if (routed.IsEmpty) return;
        Sync();
        ReadOnlySpan<int> report = reads.AsReadOnlySpan<int>();
        for (int i = 0; i < report.Length; i++) routed[i] = report[i] != 0;
    }
}
