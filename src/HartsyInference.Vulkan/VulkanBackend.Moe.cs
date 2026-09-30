using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Mixture-of-experts routing, dispatch, combine, top-k and softplus. Each follows the Core reference in
// MoeReference / TopKReference / SoftplusReference; validation is the reference's own so the errors match.
public sealed partial class VulkanBackend
{
    /// <summary>Most workgroups one dispatch may carry in x; guaranteed by the spec, so row-per-workgroup ops chunk by it.</summary>
    private const uint MaxWorkgroupsX = 65535;

    /// <inheritdoc/>
    public void Softplus(Tensor output, Tensor input)
    {
        using OpScope _op = EnterOp();
        if (input.DType != DType.F32 || output.DType != DType.F32)
        {
            throw new NotSupportedException("Vulkan Softplus supports F32 only.");
        }
        if (input.ElementCount != output.ElementCount)
        {
            throw new ArgumentException("Softplus input and output must have the same element count.");
        }
        if (input.ElementCount == 0)
        {
            return;
        }
        bool inPlace = ReferenceEquals(input, output);
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer outBuf = inPlace ? inBuf : _xfer.AllocateDevice((ulong)(input.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel k = GetKernel("softplus", storageBufferCount: 2, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)input.ElementCount);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, inBuf.Handle };
            Dispatch(k, bufs, pc.Written, GroupCount(input.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Softplus dispatch failed", ex);
            if (!inPlace) outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void MoeRoute(Tensor topkIdx, Tensor topkWeight, Tensor logits, in MoeRouteArgs args,
        Tensor? bias = null, Tensor? altBias = null, Tensor? tokenKinds = null)
    {
        MoeReference.ValidateRoute(topkIdx, topkWeight, logits, args, bias, altBias, tokenKinds);
        using OpScope _op = EnterOp();
        long tokens = logits.ElementCount / args.NumExperts;
        if (tokens > MaxWorkgroupsX)
        {
            throw new NotSupportedException($"Vulkan MoeRoute handles at most {MaxWorkgroupsX} tokens per call; got {tokens}.");
        }
        VulkanBuffer logitsBuf = GetBuffer(logits);
        // Unused optional bindings point at the logits buffer; the shader reads them only behind the has* flags.
        VulkanBuffer biasBuf = bias is null ? logitsBuf : GetBuffer(bias);
        VulkanBuffer altBuf = altBias is null ? logitsBuf : GetBuffer(altBias);
        VulkanBuffer kindsBuf = tokenKinds is null ? logitsBuf : GetBuffer(tokenKinds);
        VulkanBuffer idxBuf = _xfer.AllocateDevice((ulong)(topkIdx.ElementCount * sizeof(int)));
        VulkanBuffer wBuf = _xfer.AllocateDevice((ulong)(topkWeight.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel k = GetKernel("moe_route", storageBufferCount: 6, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)args.NumExperts);
            pc.U32((uint)args.TopK);
            pc.U32((uint)args.Scoring);
            pc.U32((uint)Math.Max(args.GroupCount, 0));
            pc.U32((uint)Math.Max(args.GroupsKept, 0));
            pc.F32(args.MaskedGroupValue);
            pc.Bool(args.Renormalize);
            pc.F32(args.RenormEpsilon);
            pc.F32(args.Scale);
            pc.F32(args.LogitDivisor);
            pc.Bool(bias is not null);
            pc.Bool(altBias is not null && tokenKinds is not null);
            Span<ulong> bufs = stackalloc ulong[]
            {
                idxBuf.Handle, wBuf.Handle, logitsBuf.Handle, biasBuf.Handle, altBuf.Handle, kindsBuf.Handle
            };
            Dispatch(k, bufs, pc.Written, (uint)tokens);
            CacheOutput(topkIdx, idxBuf);
            CacheOutput(topkWeight, wBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan MoeRoute dispatch failed", ex);
            idxBuf.Dispose();
            wBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void MoeBuildDispatch(Tensor counts, Tensor offsets, Tensor permutedToken, Tensor pairSlot, Tensor topkIdx,
        int numExperts)
    {
        MoeReference.ValidateDispatch(counts, offsets, permutedToken, pairSlot, topkIdx, numExperts);
        using OpScope _op = EnterOp();
        long pairs = topkIdx.ElementCount;
        int k = (int)topkIdx.Shape[topkIdx.Shape.Rank - 1];
        VulkanBuffer idxBuf = GetBuffer(topkIdx);
        VulkanBuffer countsBuf = _xfer.AllocateDevice((ulong)(numExperts * sizeof(int)));
        VulkanBuffer offsetsBuf = _xfer.AllocateDevice((ulong)((numExperts + 1) * sizeof(int)));
        VulkanBuffer permBuf = _xfer.AllocateDevice((ulong)Math.Max(pairs * sizeof(int), 4));
        VulkanBuffer slotBuf = _xfer.AllocateDevice((ulong)Math.Max(pairs * sizeof(int), 4));
        try
        {
            VulkanKernel kernel = GetKernel("moe_build_dispatch", storageBufferCount: 5, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)numExperts);
            pc.U32((uint)pairs);
            pc.U32((uint)k);
            Span<ulong> bufs = stackalloc ulong[]
            {
                countsBuf.Handle, offsetsBuf.Handle, permBuf.Handle, slotBuf.Handle, idxBuf.Handle
            };
            // One workgroup: the stable permutation is a sequential contract, see the shader header.
            Dispatch(kernel, bufs, pc.Written, 1);
            CacheOutput(counts, countsBuf);
            CacheOutput(offsets, offsetsBuf);
            CacheOutput(permutedToken, permBuf);
            CacheOutput(pairSlot, slotBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan MoeBuildDispatch dispatch failed", ex);
            countsBuf.Dispose();
            offsetsBuf.Dispose();
            permBuf.Dispose();
            slotBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    /// <remarks>A slot at or past the expert rows contributes nothing, where the CPU reference throws: a shader cannot raise.</remarks>
    public void MoeCombine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k, bool accumulate)
    {
        MoeReference.ValidateCombine(output, expertOut, pairSlot, topkWeight, k);
        using OpScope _op = EnterOp();
        long width = output.Shape[output.Shape.Rank - 1];
        long tokens = output.ElementCount / width;
        long expertRows = expertOut.ElementCount / width;
        if (tokens == 0)
        {
            return;
        }
        // Accumulating reads the existing rows, so upload/reuse them; otherwise a fresh buffer is enough.
        VulkanBuffer expertBuf = GetBuffer(expertOut);
        VulkanBuffer slotBuf = GetBuffer(pairSlot);
        VulkanBuffer weightBuf = GetBuffer(topkWeight);
        VulkanBuffer outBuf = accumulate ? GetBuffer(output) : _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel kernel = GetKernel("moe_combine", storageBufferCount: 4, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)tokens);
            pc.U32((uint)width);
            pc.U32((uint)k);
            pc.U32((uint)expertRows);
            pc.Bool(accumulate);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, expertBuf.Handle, slotBuf.Handle, weightBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(output.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan MoeCombine dispatch failed", ex);
            if (!accumulate) outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void TopKLastDim(Tensor values, Tensor indices, Tensor input, int k, Tensor? validLengths = null,
        bool sortByIndex = false)
    {
        TopKReference.Validate(values, indices, input, k, validLengths);
        using OpScope _op = EnterOp();
        int n = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / n;
        if (rows == 0)
        {
            return;
        }
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer validBuf = validLengths is null ? inBuf : GetBuffer(validLengths);
        VulkanBuffer valBuf = _xfer.AllocateDevice((ulong)(rows * k * sizeof(float)));
        VulkanBuffer idxBuf = _xfer.AllocateDevice((ulong)(rows * k * sizeof(int)));
        try
        {
            VulkanKernel kernel = GetKernel("topk_lastdim", storageBufferCount: 4, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Span<ulong> bufs = stackalloc ulong[] { valBuf.Handle, idxBuf.Handle, inBuf.Handle, validBuf.Handle };
            for (long first = 0; first < rows; first += MaxWorkgroupsX)
            {
                uint chunk = (uint)Math.Min(rows - first, MaxWorkgroupsX);
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)n);
                pc.U32((uint)k);
                pc.Bool(validLengths is not null);
                pc.Bool(sortByIndex);
                pc.U32((uint)first);
                Dispatch(kernel, bufs, pc.Written, chunk);
            }
            CacheOutput(values, valBuf);
            CacheOutput(indices, idxBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan TopKLastDim dispatch failed", ex);
            valBuf.Dispose();
            idxBuf.Dispose();
            throw;
        }
    }
}
