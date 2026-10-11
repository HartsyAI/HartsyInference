using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan.Authoring;

namespace HartsyInference.Vulkan;

// Expert-indexed MoE decode ops: gate/up with the activation fused, down, and the slot combine. The expert id of every routed pair is
// read from device memory, so a layer's routed stage runs from the router output to the combine with no host read. The experts of a
// projection are one device allocation (PreloadWeightGroups); the shaders address expert e at e * stride of its span buffer.
public sealed partial class VulkanBackend
{
    /// <summary>Lanes per workgroup of the expert GEMVs. A subgroup computes one output row, so a workgroup computes
    /// <c>ExpertWorkgroupLanes / subgroupSize</c> rows; enough subgroups per workgroup to keep the SM's schedulers fed without
    /// leaving a workgroup mostly idle on a small N.</summary>
    private const uint ExpertWorkgroupLanes = 128;

    private static string? MoeExpertSuffix(DType type) => type.Name switch
    {
        "Q8_0" => "q8_0",
        "Q4_K" => "q4_k",
        "Q6_K" => "q6_k",
        "Q4_0" => "q4_0",
        "Q5_0" => "q5_0",
        "Q5_K" => "q5_k",
        _ => null,
    };

    /// <summary>Output rows one workgroup of the expert GEMVs covers.</summary>
    private uint ExpertRowsPerGroup => Math.Max(1u, ExpertWorkgroupLanes / Math.Max(1u, Vk.SubgroupSize));

    /// <inheritdoc/>
    public bool SupportsMoeExpertIndexed(DType expertType)
    {
        string? suffix = MoeExpertSuffix(expertType);
        if (suffix is null || Vk.SubgroupSize is 0 or > ExpertWorkgroupLanes) return false;
        return _kernels.HasModule("moe_expert_gate_up_" + suffix) && _kernels.HasModule("moe_expert_down_" + suffix) && _kernels.HasModule("moe_combine_slots");
    }

    /// <inheritdoc/>
    public bool MoeExpertsResident(IReadOnlyList<Tensor> gateExperts, IReadOnlyList<Tensor> upExperts, IReadOnlyList<Tensor> downExperts) =>
        _xfer.TryResolveExpertGroup(gateExperts, out _, out _) && _xfer.TryResolveExpertGroup(upExperts, out _, out _)
        && _xfer.TryResolveExpertGroup(downExperts, out _, out _);

    /// <inheritdoc/>
    public void MoeExpertGateUp(Tensor act, Tensor x, IReadOnlyList<Tensor> gateExperts, IReadOnlyList<Tensor> upExperts, Tensor topkIdx,
        int topk, bool gelu)
    {
        using OpScope _op = EnterOp();
        Tensor g0 = gateExperts[0];
        DType format = g0.DType;
        string suffix = MoeExpertSuffix(format) ?? throw new NotSupportedException($"MoeExpertGateUp has no shader for {format.Name}.");
        if (upExperts[0].DType != format)
            throw new NotSupportedException($"MoeExpertGateUp has no shader for gate {format.Name} / up {upExperts[0].DType.Name}.");
        if (x.DType != DType.F32 || act.DType != DType.F32) throw new NotSupportedException("MoeExpertGateUp takes F32 activations.");
        int n = (int)g0.Shape[0], k = (int)g0.Shape[1];
        if (k % GemvBlockElements(format) != 0) throw new ArgumentException($"MoeExpertGateUp {format.Name}: K={k} is not a whole number of blocks.", nameof(x));
        int tokens = (int)(x.ElementCount / k);
        int rows = tokens * topk;
        if (act.ElementCount != (long)rows * n) throw new ArgumentException("act must hold tokens * topk rows of N.", nameof(act));
        if (topkIdx.ElementCount != rows) throw new ArgumentException("topkIdx must hold tokens * topk ids.", nameof(topkIdx));
        if (!_xfer.TryResolveExpertGroup(gateExperts, out VulkanBuffer? gateSpan, out long stride)
            || !_xfer.TryResolveExpertGroup(upExperts, out VulkanBuffer? upSpan, out long upStride) || upStride != stride)
            throw new InvalidOperationException("MoeExpertGateUp needs the gate and up experts resident and contiguous.");

        uint groupsX = ExpertGroupsX(n);
        VulkanBuffer xBuf = GetBuffer(x);
        VulkanBuffer idxBuf = GetBuffer(topkIdx);
        VulkanBuffer actBuf = _xfer.AllocateDevice((ulong)((long)rows * n * sizeof(float)));
        try
        {
            VulkanKernel kernel = GetKernel("moe_expert_gate_up_" + suffix, storageBufferCount: 5, new SpecConstant[]
            {
                SpecConstant.UInt(0, ExpertWorkgroupLanes),
                SpecConstant.Bool(1, gelu),
            });
            Span<byte> raw = stackalloc byte[24];
            Span<ulong> bufs = stackalloc ulong[] { gateSpan!.Handle, upSpan!.Handle, xBuf.Handle, idxBuf.Handle, actBuf.Handle };
            for (long first = 0; first < rows; first += MaxWorkgroupsX)
            {
                PushConstants pc = new(raw);
                pc.U32((uint)n);
                pc.U32((uint)k);
                pc.U32((uint)topk);
                pc.U32((uint)first);
                pc.U32((uint)stride);
                pc.U32((uint)gateExperts.Count);
                Dispatch(kernel, bufs, pc.Written, groupsX, (uint)Math.Min(rows - first, MaxWorkgroupsX), 1);
            }
            CacheOutput(act, actBuf);
        }
        catch
        {
            actBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void MoeExpertDown(Tensor slotOut, Tensor act, IReadOnlyList<Tensor> downExperts, Tensor topkIdx, int topk)
    {
        using OpScope _op = EnterOp();
        Tensor d0 = downExperts[0];
        DType format = d0.DType;
        string suffix = MoeExpertSuffix(format) ?? throw new NotSupportedException($"MoeExpertDown has no shader for {format.Name}.");
        if (act.DType != DType.F32 || slotOut.DType != DType.F32) throw new NotSupportedException("MoeExpertDown takes F32 activations.");
        int n = (int)d0.Shape[0], k = (int)d0.Shape[1];
        if (k % GemvBlockElements(format) != 0) throw new ArgumentException($"MoeExpertDown {format.Name}: K={k} is not a whole number of blocks.", nameof(act));
        int rows = (int)(act.ElementCount / k);
        if (slotOut.ElementCount != (long)rows * n) throw new ArgumentException("slotOut must hold one row of N per act row.", nameof(slotOut));
        if (topkIdx.ElementCount != rows) throw new ArgumentException("topkIdx must hold one id per act row.", nameof(topkIdx));
        if (!_xfer.TryResolveExpertGroup(downExperts, out VulkanBuffer? downSpan, out long stride))
            throw new InvalidOperationException("MoeExpertDown needs the down experts resident and contiguous.");

        uint groupsX = ExpertGroupsX(n);
        VulkanBuffer actBuf = GetBuffer(act);
        VulkanBuffer idxBuf = GetBuffer(topkIdx);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)((long)rows * n * sizeof(float)));
        try
        {
            VulkanKernel kernel = GetKernel("moe_expert_down_" + suffix, storageBufferCount: 4, new SpecConstant[]
            {
                SpecConstant.UInt(0, ExpertWorkgroupLanes),
            });
            Span<byte> raw = stackalloc byte[20];
            Span<ulong> bufs = stackalloc ulong[] { downSpan!.Handle, actBuf.Handle, idxBuf.Handle, outBuf.Handle };
            for (long first = 0; first < rows; first += MaxWorkgroupsX)
            {
                PushConstants pc = new(raw);
                pc.U32((uint)n);
                pc.U32((uint)k);
                pc.U32((uint)first);
                pc.U32((uint)stride);
                pc.U32((uint)downExperts.Count);
                Dispatch(kernel, bufs, pc.Written, groupsX, (uint)Math.Min(rows - first, MaxWorkgroupsX), 1);
            }
            CacheOutput(slotOut, outBuf);
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }

    /// <summary>Workgroups along x to cover <paramref name="n"/> output rows, one subgroup per row.</summary>
    private uint ExpertGroupsX(int n)
    {
        uint rowsPerGroup = ExpertRowsPerGroup;
        long groups = (n + rowsPerGroup - 1) / rowsPerGroup;
        if (groups > MaxWorkgroupsX) throw new NotSupportedException($"An expert projection of {n} output rows needs {groups} workgroups; the limit is {MaxWorkgroupsX}.");
        return (uint)groups;
    }

    /// <inheritdoc/>
    public void MoeCombineSlots(Tensor output, Tensor slotOut, Tensor topkWeight, Tensor? shared, Tensor? sharedGateLogit, int topk)
    {
        using OpScope _op = EnterOp();
        if (output.DType != DType.F32 || slotOut.DType != DType.F32 || topkWeight.DType != DType.F32)
            throw new NotSupportedException("MoeCombineSlots takes F32 tensors.");
        int hidden = (int)output.Shape[output.Shape.Rank - 1];
        long tokens = output.ElementCount / hidden;
        if (slotOut.ElementCount != tokens * topk * hidden)
            throw new ArgumentException("slotOut must hold topk rows of hidden per token.", nameof(slotOut));
        if (topkWeight.ElementCount != tokens * topk) throw new ArgumentException("topkWeight must hold topk weights per token.", nameof(topkWeight));
        if (shared is not null && shared.ElementCount != output.ElementCount) throw new ArgumentException("shared must match the output.", nameof(shared));
        if (sharedGateLogit is not null && (shared is null || sharedGateLogit.ElementCount != tokens))
            throw new ArgumentException("sharedGateLogit needs a shared row and one logit per token.", nameof(sharedGateLogit));
        if (tokens == 0) return;

        VulkanBuffer slotBuf = GetBuffer(slotOut);
        VulkanBuffer weightBuf = GetBuffer(topkWeight);
        VulkanBuffer sharedBuf = shared is null ? slotBuf : GetBuffer(shared);
        VulkanBuffer gateBuf = sharedGateLogit is null ? slotBuf : GetBuffer(sharedGateLogit);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel kernel = GetKernel("moe_combine_slots", storageBufferCount: 5, ReadOnlySpan<SpecConstant>.Empty);
            Span<byte> raw = stackalloc byte[20];
            PushConstants pc = new(raw);
            pc.U32((uint)output.ElementCount);
            pc.U32((uint)hidden);
            pc.U32((uint)topk);
            pc.U32(shared is null ? 0u : 1u);
            pc.U32(sharedGateLogit is null ? 0u : 1u);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, slotBuf.Handle, weightBuf.Handle, sharedBuf.Handle, gateBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    /// <remarks>A projection's experts become one allocation when they are equal-shaped members of a format the expert GEMVs read
    /// (<see cref="SupportsMoeExpertIndexed"/> then needs <see cref="MoeExpertsResident"/> to hold); every other group, and a group the
    /// device cannot place that way, is uploaded member by member as before.</remarks>
    protected override void PreloadExpandedWeightGroups(IEnumerable<IReadOnlyList<Tensor>> groups)
    {
        foreach (IReadOnlyList<Tensor> group in groups)
        {
            if (group.Count > 1 && MoeExpertSuffix(group[0].DType) is not null && _xfer.TryPreloadWeightGroup(group))
            {
                continue;
            }
            PreloadExpandedWeights(group);
        }
    }
}
