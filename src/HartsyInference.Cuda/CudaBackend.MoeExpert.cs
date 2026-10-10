using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Expert-indexed MoE decode ops: gate/up with the activation fused, down, and the slot combine. The expert id of every routed
// pair is read from device memory, so a layer's routed stage runs from the router output to the combine without a host read
// and inside a captured graph.
public sealed partial class CudaBackend
{
    /// <inheritdoc />
    public bool SupportsMoeExpertIndexed(DType expertType)
    {
        if (_kernels is null) return false;
        return _kernels.HasMoeExpertKernel(expertType) && _kernels.HasMoeCombineSlots;
    }

    /// <inheritdoc />
    public void MoeExpertGateUp(Tensor act, Tensor x, IReadOnlyList<Tensor> gateExperts, IReadOnlyList<Tensor> upExperts,
        Tensor topkIdx, int topk, bool gelu)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeExpertGateUp");
        using OpScope _op = EnterOp();
        EnsureKernels();
        Tensor g0 = gateExperts[0];
        DType format = g0.DType;
        if (!_kernels!.HasMoeExpertKernel(format) || upExperts[0].DType != format)
            throw new NotSupportedException($"MoeExpertGateUp has no kernel for gate {format} / up {upExperts[0].DType}.");
        int n = (int)g0.Shape[0], k = (int)g0.Shape[1];
        int tokens = (int)(x.ElementCount / k);
        int rows = tokens * topk;
        if (act.ElementCount != (long)rows * n) throw new ArgumentException("act must hold tokens * topk rows of N.", nameof(act));
        if (topkIdx.ElementCount != rows) throw new ArgumentException("topkIdx must hold tokens * topk ids.", nameof(topkIdx));
        if (!TryResolveExpertGroup(gateExperts, out ulong gateBase, out long stride)
            || !TryResolveExpertGroup(upExperts, out ulong upBase, out long upStride) || upStride != stride)
            throw new InvalidOperationException("MoeExpertGateUp needs the gate and up experts resident and contiguous.");

        ulong pX = 0, pIdx = 0, pAct = 0;
        bool cachedAct = false;
        (ulong xq, ulong xd, ulong xs, bool transient) q = default;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pIdx = GpuTransferHelper.CopyToDevice(topkIdx);
            nuint actBytes = GpuTransferHelper.ByteSize(act);
            pAct = GpuTransferHelper.AllocateDevice(actBytes);
            q = PrepareActivationQ8(x, pX, tokens, k);
            _kernels.LaunchMoeGateUpId(format, pAct, q.xq, q.xd, q.xs, gateBase, upBase, pIdx, stride, n, k, topk, rows,
                gateExperts.Count, gelu, _stream.Handle);
            GpuTransferHelper.CacheActivation(act, pAct, actBytes);
            cachedAct = true;
        }
        finally
        {
            ReleaseActivationQ8(q);
            if (!cachedAct) GpuTransferHelper.FreeDevice(pAct);
            GpuTransferHelper.FreeDevice(pIdx);
            GpuTransferHelper.FreeDevice(pX);
        }
    }

    /// <inheritdoc />
    public void MoeExpertDown(Tensor slotOut, Tensor act, IReadOnlyList<Tensor> downExperts, Tensor topkIdx, int topk)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeExpertDown");
        using OpScope _op = EnterOp();
        EnsureKernels();
        Tensor d0 = downExperts[0];
        DType format = d0.DType;
        if (!_kernels!.HasMoeExpertKernel(format))
            throw new NotSupportedException($"MoeExpertDown has no kernel for {format}.");
        int n = (int)d0.Shape[0], k = (int)d0.Shape[1];
        int rows = (int)(act.ElementCount / k);
        if (slotOut.ElementCount != (long)rows * n) throw new ArgumentException("slotOut must hold one row of N per act row.", nameof(slotOut));
        if (topkIdx.ElementCount != rows) throw new ArgumentException("topkIdx must hold one id per act row.", nameof(topkIdx));
        if (!TryResolveExpertGroup(downExperts, out ulong downBase, out long stride))
            throw new InvalidOperationException("MoeExpertDown needs the down experts resident and contiguous.");

        ulong pAct = 0, pIdx = 0, pOut = 0;
        bool cachedOut = false;
        (ulong xq, ulong xd, ulong xs, bool transient) q = default;
        try
        {
            pAct = GpuTransferHelper.CopyToDevice(act);
            pIdx = GpuTransferHelper.CopyToDevice(topkIdx);
            nuint outBytes = GpuTransferHelper.ByteSize(slotOut);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            q = PrepareActivationQ8(act, pAct, rows, k);
            _kernels.LaunchMoeDownId(format, pOut, q.xq, q.xd, q.xs, downBase, pIdx, stride, n, k, rows, downExperts.Count,
                _stream.Handle);
            GpuTransferHelper.CacheActivation(slotOut, pOut, outBytes);
            cachedOut = true;
        }
        finally
        {
            ReleaseActivationQ8(q);
            if (!cachedOut) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIdx);
            GpuTransferHelper.FreeDevice(pAct);
        }
    }

    /// <inheritdoc />
    public void MoeCombineSlots(Tensor output, Tensor slotOut, Tensor topkWeight, Tensor? shared, Tensor? sharedGateLogit, int topk)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeCombineSlots");
        using OpScope _op = EnterOp();
        EnsureKernels();
        RequireMoeKernels();
        int hidden = (int)output.Shape[output.Shape.Rank - 1];
        int tokens = (int)(output.ElementCount / hidden);
        if (slotOut.ElementCount != (long)tokens * topk * hidden)
            throw new ArgumentException("slotOut must hold topk rows of hidden per token.", nameof(slotOut));
        if (topkWeight.ElementCount != (long)tokens * topk) throw new ArgumentException("topkWeight must hold topk weights per token.", nameof(topkWeight));
        ulong pOut = 0, pSlots = 0, pW = 0, pShared = 0, pGate = 0;
        bool cached = false;
        try
        {
            pSlots = GpuTransferHelper.CopyToDevice(slotOut);
            pW = GpuTransferHelper.CopyToDevice(topkWeight);
            if (shared is not null) pShared = GpuTransferHelper.CopyToDevice(shared);
            if (sharedGateLogit is not null) pGate = GpuTransferHelper.CopyToDevice(sharedGateLogit);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchMoeCombineSlots(pOut, pSlots, pW, pShared, pGate, tokens, hidden, topk, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pSlots);
            GpuTransferHelper.FreeDevice(pW);
            if (pShared != 0) GpuTransferHelper.FreeDevice(pShared);
            if (pGate != 0) GpuTransferHelper.FreeDevice(pGate);
        }
    }

    /// <summary>Resolves a group of per-expert weights to its base device address and per-expert byte stride. The experts must be
    /// resident weights placed back to back (<see cref="IBackend.PreloadWeightGroups"/> does that). The first, middle and last
    /// member are checked against the stride, so a group whose members were re-placed individually is refused rather than read at
    /// a wrong address.</summary>
    private bool TryResolveExpertGroup(IReadOnlyList<Tensor> experts, out ulong baseAddr, out long stride)
    {
        baseAddr = 0;
        stride = 0;
        int count = experts.Count;
        if (count == 0) return false;
        Tensor first = experts[0];
        if (!GpuTransferHelper.IsWeightCached(first)) return false;
        ulong p0 = GpuTransferHelper.CopyToDevice(first);
        if (count == 1)
        {
            baseAddr = p0;
            stride = (long)GpuTransferHelper.ByteSize(first);
            return true;
        }
        Tensor second = experts[1];
        if (!GpuTransferHelper.IsWeightCached(second)) return false;
        ulong p1 = GpuTransferHelper.CopyToDevice(second);
        if (p1 <= p0) return false;
        long step = (long)(p1 - p0);
        if (step < (long)GpuTransferHelper.ByteSize(first)) return false;
        foreach (int i in new[] { count / 2, count - 1 })
        {
            Tensor member = experts[i];
            if (!GpuTransferHelper.IsWeightCached(member)) return false;
            if (GpuTransferHelper.CopyToDevice(member) != p0 + (ulong)(step * i)) return false;
        }
        baseAddr = p0;
        stride = step;
        return true;
    }

    /// <summary>The Q8_1 form (int8 values, per-32 scale and sum) of <paramref name="rows"/> activation rows of length
    /// <paramref name="k"/>: the producer's sidecar when a single row carries one, else quantized into the persistent scratch, or into
    /// transient buffers when the scratch would have to grow while the stream is being captured.</summary>
    private (ulong xq, ulong xd, ulong xs, bool transient) PrepareActivationQ8(Tensor x, ulong pX, int rows, int k)
    {
        if (rows == 1 && GpuTransferHelper.TryGetSidecar(x, k, out ulong scXq, out ulong scXd, out ulong scXs))
            return (scXq, scXd, scXs, false);
        long blocks = (long)rows * (k / 32);
        nuint xqBytes = (nuint)(((long)rows * k + 255) & ~255L);
        nuint xdBytes = (nuint)((blocks * sizeof(float) + 255) & ~255L);
        ulong scratch = EnsureDp4aScratch(xqBytes + 2 * xdBytes);
        bool transient = scratch == 0;
        ulong xq = transient ? GpuTransferHelper.AllocateDevice((nuint)((long)rows * k)) : scratch;
        ulong xd = transient ? GpuTransferHelper.AllocateDevice((nuint)(blocks * sizeof(float))) : scratch + xqBytes;
        ulong xs = transient ? GpuTransferHelper.AllocateDevice((nuint)(blocks * sizeof(float))) : scratch + xqBytes + xdBytes;
        _kernels!.LaunchQuantizeActivationQ8_1(xq, xd, xs, pX, rows, k, _stream.Handle);
        return (xq, xd, xs, transient);
    }

    private static void ReleaseActivationQ8((ulong xq, ulong xd, ulong xs, bool transient) q)
    {
        if (!q.transient) return;
        GpuTransferHelper.FreeDevice(q.xq);
        GpuTransferHelper.FreeDevice(q.xd);
        GpuTransferHelper.FreeDevice(q.xs);
    }
}
