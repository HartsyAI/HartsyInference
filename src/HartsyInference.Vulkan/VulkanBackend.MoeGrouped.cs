using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan.Authoring;

namespace HartsyInference.Vulkan;

// Device-routed MoE prefill. MoeFeedForward routes and permutes on the device (MoeRoute, MoeBuildDispatch) and reads the E + 1 offsets once per
// layer; these ops then run the per-expert GEMMs over contiguous row ranges and fold the pair map back into token order. Only the experts that
// received rows are dequantized, one at a time into transient F16 weights (a layer's whole stack as F16 is about 1 GB on OLMoE, more than a
// 12 GB card can spare), and the GEMMs are the cooperative-matrix ones the dense prefill uses, with F32 accumulation and F32 outputs.
public sealed partial class VulkanBackend
{
    /// <inheritdoc/>
    public bool SupportsMoeExpertsGrouped(DType expertType)
    {
        // The per-expert GEMM writes an F32 output at a row offset, which only the cooperative-matrix kernels do; the register-tiled fallback
        // would round it through F16. A device without them keeps the host-routed layer rather than a lossy device path.
        if (MoeExpertSuffix(expertType) is null || !GroupedGemmAvailable) return false;
        return _kernels.HasModule("moe_gather_rows_f16") && _kernels.HasModule("moe_act_mul_f16") && _kernels.HasModule("moe_combine_pairs");
    }

    private bool GroupedGemmAvailable => (EnableCoopMat2 && Vk.HasCooperativeMatrix2) || (!_disableCoopmat && Vk.HasCooperativeMatrix);

    /// <inheritdoc/>
    public void MoeExpertsGrouped(Tensor expertOut, Tensor x, Tensor permutedToken, ReadOnlySpan<int> offsets,
        IReadOnlyList<Tensor> gateExperts, IReadOnlyList<Tensor> upExperts, IReadOnlyList<Tensor> downExperts, bool gelu)
    {
        using OpScope _op = EnterOp();
        int experts = gateExperts.Count;
        if (offsets.Length != experts + 1) throw new ArgumentException("offsets must hold experts + 1 entries.", nameof(offsets));
        if (upExperts.Count != experts || downExperts.Count != experts) throw new ArgumentException("gate, up and down need the same number of experts.");
        Tensor g0 = gateExperts[0], d0 = downExperts[0];
        if (!SupportsMoeExpertsGrouped(g0.DType) || !SupportsMoeExpertsGrouped(d0.DType) || upExperts[0].DType != g0.DType)
            throw new NotSupportedException($"MoeExpertsGrouped has no path for gate {g0.DType.Name} / down {d0.DType.Name}.");
        if (x.DType != DType.F32 || expertOut.DType != DType.F32) throw new NotSupportedException("MoeExpertsGrouped takes F32 activations.");
        int inter = (int)g0.Shape[0], hidden = (int)g0.Shape[1];
        if (d0.Shape[0] != hidden || d0.Shape[1] != inter) throw new ArgumentException("The down experts must be [hidden, inter].", nameof(downExperts));
        if (inter % 16 != 0 || hidden % 16 != 0) throw new NotSupportedException($"MoeExpertsGrouped needs hidden ({hidden}) and inter ({inter}) in multiples of 16.");
        int rows = offsets[experts];
        if (expertOut.ElementCount != (long)rows * hidden) throw new ArgumentException("expertOut must hold one row of hidden per routed pair.", nameof(expertOut));
        long tokens = x.ElementCount / hidden;
        if (permutedToken.ElementCount < rows) throw new ArgumentException("permutedToken must hold a token id per routed row.", nameof(permutedToken));
        if ((long)rows * hidden > uint.MaxValue || (long)rows * inter > uint.MaxValue)
            throw new NotSupportedException($"MoeExpertsGrouped: {rows} routed rows overflow the 32-bit GEMM offsets; prefill in smaller chunks.");
        int maxCount = 0;
        for (int e = 0; e < experts; e++)
        {
            if (offsets[e + 1] < offsets[e]) throw new ArgumentException("offsets must be non-decreasing.", nameof(offsets));
            maxCount = Math.Max(maxCount, offsets[e + 1] - offsets[e]);
        }

        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)Math.Max((long)rows * hidden * sizeof(float), 4));
        VulkanBuffer? permX = null, gateOut = null, upOut = null, act = null;
        try
        {
            if (rows > 0)
            {
                VulkanBuffer xBuf = GetBuffer(x);
                VulkanBuffer permBuf = GetBuffer(permutedToken);
                permX = _xfer.AllocateDevice((ulong)((long)rows * hidden * sizeof(ushort)));
                gateOut = _xfer.AllocateDevice((ulong)((long)maxCount * inter * sizeof(float)));
                upOut = _xfer.AllocateDevice((ulong)((long)maxCount * inter * sizeof(float)));
                act = _xfer.AllocateDevice((ulong)((long)maxCount * inter * sizeof(ushort)));

                long gatherTotal = (long)rows * hidden;
                VulkanKernel gather = GetKernel("moe_gather_rows_f16", storageBufferCount: 3, _default1DSpec);
                Span<byte> raw = stackalloc byte[12];
                PushConstants gpc = new(raw);
                gpc.U32((uint)rows);
                gpc.U32((uint)hidden);
                gpc.U32((uint)tokens);
                Span<ulong> gbufs = stackalloc ulong[] { permX.Handle, xBuf.Handle, permBuf.Handle };
                Dispatch(gather, gbufs, gpc.Written, GroupCount(gatherTotal, LocalX1D));

                Span<byte> apcRaw = stackalloc byte[8];
                Span<ulong> abufs = stackalloc ulong[] { act.Handle, gateOut.Handle, upOut.Handle };
                VulkanKernel actMul = GetKernel("moe_act_mul_f16", storageBufferCount: 3, _default1DSpec);
                for (int e = 0; e < experts; e++)
                {
                    int first = offsets[e], count = offsets[e + 1] - first;
                    if (count == 0) continue;
                    // Dequantize this expert's three projections to F16 (cached or transient per the cast policy; transients are freed deferred).
                    (VulkanBuffer wGate, VulkanBuffer? ownGate) = CastIfNeeded(gateExperts[e], GetBuffer(gateExperts[e]), DType.F16);
                    (VulkanBuffer wUp, VulkanBuffer? ownUp) = CastIfNeeded(upExperts[e], GetBuffer(upExperts[e]), DType.F16);
                    (VulkanBuffer wDown, VulkanBuffer? ownDown) = CastIfNeeded(downExperts[e], GetBuffer(downExperts[e]), DType.F16);
                    try
                    {
                        uint aOffset = (uint)((long)first * hidden);
                        // C[count, inter] = permX[first.., hidden] . W[inter, hidden]^T, F32 accumulate and F32 out.
                        DispatchGemm(new GemmOperands(permX.Handle, wGate.Handle, gateOut.Handle, count, inter, hidden, false, true, DType.F16, DType.F32)
                        { AOffset = aOffset });
                        DispatchGemm(new GemmOperands(permX.Handle, wUp.Handle, upOut.Handle, count, inter, hidden, false, true, DType.F16, DType.F32)
                        { AOffset = aOffset });

                        long actCount = (long)count * inter;
                        PushConstants apc = new(apcRaw);
                        apc.U32((uint)actCount);
                        apc.U32(gelu ? 1u : 0u);
                        Dispatch(actMul, abufs, apc.Written, GroupCount(actCount, LocalX1D));

                        // expertOut[first.., hidden] = act[count, inter] . Wdown[hidden, inter]^T
                        DispatchGemm(new GemmOperands(act.Handle, wDown.Handle, outBuf.Handle, count, hidden, inter, false, true, DType.F16, DType.F32)
                        { COffset = (uint)((long)first * hidden) });
                    }
                    finally
                    {
                        if (ownGate is not null) _xfer.FreeDevice(ownGate);
                        if (ownUp is not null) _xfer.FreeDevice(ownUp);
                        if (ownDown is not null) _xfer.FreeDevice(ownDown);
                    }
                }
            }
            CacheOutput(expertOut, outBuf);
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
        finally
        {
            // The scratch is read by dispatches already recorded; the deferred free holds it until they have run.
            if (permX is not null) _xfer.FreeDevice(permX);
            if (gateOut is not null) _xfer.FreeDevice(gateOut);
            if (upOut is not null) _xfer.FreeDevice(upOut);
            if (act is not null) _xfer.FreeDevice(act);
        }
    }

    /// <inheritdoc/>
    public void MoeCombinePairs(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, Tensor? shared, Tensor? sharedGateLogit, int topk)
    {
        using OpScope _op = EnterOp();
        if (output.DType != DType.F32 || expertOut.DType != DType.F32 || topkWeight.DType != DType.F32)
            throw new NotSupportedException("MoeCombinePairs takes F32 tensors.");
        int hidden = (int)output.Shape[output.Shape.Rank - 1];
        long tokens = output.ElementCount / hidden;
        if (pairSlot.ElementCount != tokens * topk) throw new ArgumentException("pairSlot must hold topk slots per token.", nameof(pairSlot));
        if (topkWeight.ElementCount != tokens * topk) throw new ArgumentException("topkWeight must hold topk weights per token.", nameof(topkWeight));
        if (expertOut.ElementCount % hidden != 0) throw new ArgumentException("expertOut must be rows of hidden.", nameof(expertOut));
        if (shared is not null && shared.ElementCount != output.ElementCount) throw new ArgumentException("shared must match the output.", nameof(shared));
        if (sharedGateLogit is not null && (shared is null || sharedGateLogit.ElementCount != tokens))
            throw new ArgumentException("sharedGateLogit needs a shared row and one logit per token.", nameof(sharedGateLogit));
        if (tokens == 0) return;

        long expertRows = expertOut.ElementCount / hidden;
        VulkanBuffer expertBuf = GetBuffer(expertOut);
        VulkanBuffer slotBuf = GetBuffer(pairSlot);
        VulkanBuffer weightBuf = GetBuffer(topkWeight);
        VulkanBuffer sharedBuf = shared is null ? expertBuf : GetBuffer(shared);
        VulkanBuffer gateBuf = sharedGateLogit is null ? expertBuf : GetBuffer(sharedGateLogit);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel kernel = GetKernel("moe_combine_pairs", storageBufferCount: 6, ReadOnlySpan<SpecConstant>.Empty);
            Span<byte> raw = stackalloc byte[24];
            PushConstants pc = new(raw);
            pc.U32((uint)output.ElementCount);
            pc.U32((uint)hidden);
            pc.U32((uint)topk);
            pc.U32((uint)expertRows);
            pc.U32(shared is null ? 0u : 1u);
            pc.U32(sharedGateLogit is null ? 0u : 1u);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, expertBuf.Handle, slotBuf.Handle, weightBuf.Handle, sharedBuf.Handle, gateBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }
}
