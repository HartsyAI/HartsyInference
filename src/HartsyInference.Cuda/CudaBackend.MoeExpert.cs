using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

// Expert-indexed MoE decode ops: gate/up with the activation fused, down, and the slot combine. The expert id of every routed
// pair is read from device memory, so a layer's routed stage runs from the router output to the combine without a host read
// and inside a captured graph.
public sealed unsafe partial class CudaBackend
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

    /// <inheritdoc />
    public bool SupportsMoeExpertsGrouped(DType expertType)
    {
        if (_kernels is null || !_kernels.HasMoePrefillKernels || !_kernels.HasMoeKernels) return false;
        if (!expertType.IsQuantized || !_kernels.GgufDequantTypes.Contains(expertType)) return false;
        DType gemm = ResolveGemmDtype(DType.F32, expertType);
        return gemm == DType.BF16 || gemm == DType.F16;
    }

    /// <inheritdoc />
    public void MoeExpertsGrouped(Tensor expertOut, Tensor x, Tensor permutedToken, ReadOnlySpan<int> offsets,
        IReadOnlyList<Tensor> gateExperts, IReadOnlyList<Tensor> upExperts, IReadOnlyList<Tensor> downExperts, bool gelu)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeExpertsGrouped");
        using OpScope _op = EnterOp();
        EnsureKernels();
        int experts = gateExperts.Count;
        if (offsets.Length != experts + 1) throw new ArgumentException("offsets must hold experts + 1 entries.", nameof(offsets));
        Tensor g0 = gateExperts[0], u0 = upExperts[0], d0 = downExperts[0];
        DType gemmDtype = ResolveGemmDtype(DType.F32, g0.DType);
        if (!SupportsMoeExpertsGrouped(g0.DType) || !SupportsMoeExpertsGrouped(d0.DType) || u0.DType != g0.DType
            || ResolveGemmDtype(DType.F32, d0.DType) != gemmDtype)
            throw new NotSupportedException($"MoeExpertsGrouped has no path for gate {g0.DType} / down {d0.DType}.");
        int inter = (int)g0.Shape[0], hidden = (int)g0.Shape[1];
        int rows = offsets[experts];
        if (expertOut.ElementCount != (long)rows * hidden) throw new ArgumentException("expertOut must hold one row of hidden per routed pair.", nameof(expertOut));
        int maxCount = 0;
        for (int e = 0; e < experts; e++) maxCount = Math.Max(maxCount, offsets[e + 1] - offsets[e]);
        bool bf16 = gemmDtype == DType.BF16;
        int gemmType = CublasApi.DataTypeOf(gemmDtype);
        int elem = gemmDtype.SizeInBytes;

        ulong pX = 0, pPerm = 0, pOut = 0, permX = 0, gateOut = 0, upOut = 0, act = 0, wsGate = 0, wsUp = 0, wsDown = 0;
        bool cachedOut = false;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pPerm = GpuTransferHelper.CopyToDevice(permutedToken);
            nuint outBytes = GpuTransferHelper.ByteSize(expertOut);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (rows > 0)
            {
                permX = GpuTransferHelper.AllocateDevice((nuint)((long)rows * hidden * elem));
                gateOut = GpuTransferHelper.AllocateDevice((nuint)((long)maxCount * inter * sizeof(float)));
                upOut = GpuTransferHelper.AllocateDevice((nuint)((long)maxCount * inter * sizeof(float)));
                act = GpuTransferHelper.AllocateDevice((nuint)((long)maxCount * inter * elem));
                long wsBytes = (long)inter * hidden * elem;
                wsGate = GpuTransferHelper.AllocateDevice((nuint)wsBytes);
                wsUp = GpuTransferHelper.AllocateDevice((nuint)wsBytes);
                wsDown = GpuTransferHelper.AllocateDevice((nuint)wsBytes);
                _kernels!.LaunchMoeGatherRows16(permX, pX, pPerm, rows, hidden, bf16, _stream.Handle);

                float alpha = 1f, beta = 0f;
                int compute = Compute32F(gemmType);
                for (int e = 0; e < experts; e++)
                {
                    int first = offsets[e], count = offsets[e + 1] - first;
                    if (count == 0) continue;
                    ulong pG = GpuTransferHelper.CopyToDevice(gateExperts[e]);
                    ulong pU = GpuTransferHelper.CopyToDevice(upExperts[e]);
                    ulong pD = GpuTransferHelper.CopyToDevice(downExperts[e]);
                    try
                    {
                        ulong wG = ExpertWeightCast(gateExperts[e], pG, gemmDtype, wsGate);
                        ulong wU = ExpertWeightCast(upExperts[e], pU, gemmDtype, wsUp);
                        ulong wD = ExpertWeightCast(downExperts[e], pD, gemmDtype, wsDown);
                        ulong a = permX + (ulong)((long)first * hidden * elem);
                        // Row-major out[count, n] = a[count, k] x w[n, k]^T is column-major out^T[n, count] = w^T x a^T.
                        CublasApi.cublasGemmEx(_cublasHandle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N, inter, count, hidden, &alpha,
                            wG, gemmType, hidden, a, gemmType, hidden, &beta, gateOut, CublasApi.CUDA_R_32F, inter, compute,
                            CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                        CublasApi.cublasGemmEx(_cublasHandle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N, inter, count, hidden, &alpha,
                            wU, gemmType, hidden, a, gemmType, hidden, &beta, upOut, CublasApi.CUDA_R_32F, inter, compute,
                            CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                        _kernels.LaunchMoeActMul16(act, gateOut, upOut, (long)count * inter, gelu, bf16, _stream.Handle);
                        ulong dst = pOut + (ulong)((long)first * hidden * sizeof(float));
                        CublasApi.cublasGemmEx(_cublasHandle, CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N, hidden, count, inter, &alpha,
                            wD, gemmType, inter, act, gemmType, inter, &beta, dst, CublasApi.CUDA_R_32F, hidden, compute,
                            CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                    }
                    finally
                    {
                        GpuTransferHelper.FreeDevice(pG);
                        GpuTransferHelper.FreeDevice(pU);
                        GpuTransferHelper.FreeDevice(pD);
                    }
                }
            }
            GpuTransferHelper.CacheActivation(expertOut, pOut, outBytes);
            cachedOut = true;
        }
        finally
        {
            if (!cachedOut) GpuTransferHelper.FreeDevice(pOut);
            foreach (ulong p in new[] { permX, gateOut, upOut, act, wsGate, wsUp, wsDown })
                if (p != 0) GpuTransferHelper.FreeDevice(p);
            GpuTransferHelper.FreeDevice(pPerm);
            GpuTransferHelper.FreeDevice(pX);
        }
    }

    /// <summary>The 16-bit form of an expert weight for the GEMM: the cached cast when one exists, a newly cached one while free
    /// memory stays above the quantized-weight headroom (the budget gate <c>Linear</c> applies), else a transient cast into
    /// <paramref name="workspace"/>.</summary>
    private ulong ExpertWeightCast(Tensor weight, ulong deviceWeight, DType gemmDtype, ulong workspace)
    {
        if (CacheWeightCasts && GpuTransferHelper.IsWeightCached(weight))
        {
            if (GpuTransferHelper.TryGetWeightCast(weight, gemmDtype, out ulong cached)) return cached;
            nuint castBytes = (nuint)(weight.ElementCount * gemmDtype.SizeInBytes);
            (long freeBytes, long totalBytes) = CudaMemory.GetMemInfo();
            long headroom = Math.Max(4L << 30, totalBytes / 3);
            if (freeBytes <= 0 || freeBytes - (long)castBytes >= headroom)
            {
                ulong cast = GpuTransferHelper.AllocateDevice(castBytes);
                CastOnGpu(cast, deviceWeight, weight.DType, gemmDtype, (int)weight.ElementCount);
                GpuTransferHelper.CacheWeightCast(weight, gemmDtype, cast, castBytes);
                return cast;
            }
        }
        CastOnGpu(workspace, deviceWeight, weight.DType, gemmDtype, (int)weight.ElementCount);
        return workspace;
    }

    /// <inheritdoc />
    public void MoeCombinePairs(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, Tensor? shared,
        Tensor? sharedGateLogit, int topk)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeCombinePairs");
        using OpScope _op = EnterOp();
        EnsureKernels();
        int hidden = (int)output.Shape[output.Shape.Rank - 1];
        int tokens = (int)(output.ElementCount / hidden);
        int expertRows = (int)(expertOut.ElementCount / hidden);
        ulong pOut = 0, pExp = 0, pSlot = 0, pW = 0, pShared = 0, pGate = 0;
        bool cached = false;
        try
        {
            pExp = GpuTransferHelper.CopyToDevice(expertOut);
            pSlot = GpuTransferHelper.CopyToDevice(pairSlot);
            pW = GpuTransferHelper.CopyToDevice(topkWeight);
            if (shared is not null) pShared = GpuTransferHelper.CopyToDevice(shared);
            if (sharedGateLogit is not null) pGate = GpuTransferHelper.CopyToDevice(sharedGateLogit);
            nuint bytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchMoeCombinePairs(pOut, pExp, pSlot, pW, pShared, pGate, tokens, hidden, topk, expertRows, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pExp);
            GpuTransferHelper.FreeDevice(pSlot);
            GpuTransferHelper.FreeDevice(pW);
            if (pShared != 0) GpuTransferHelper.FreeDevice(pShared);
            if (pGate != 0) GpuTransferHelper.FreeDevice(pGate);
        }
    }
}
