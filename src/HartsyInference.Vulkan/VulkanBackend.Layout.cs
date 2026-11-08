using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Generic layout moves: strided copy, strided accumulate and pad (copy_strided, accumulate_strided, pad_strided
// shaders), validated through the Core contracts. The shaders address 16, 32 and 64-bit words with 32-bit
// indices; 8-bit elements and BF16 accumulation take the host reference, as the interface defaults did.
public sealed partial class VulkanBackend
{
    /// <inheritdoc />
    public void CopyStrided(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        using OpScope _op = EnterOp();
        StridedCopyGeometry g = StridedCopyContract.ValidateCopy(output, input, spec);
        if (g.ElementSize == 1)
        {
            StridedCopyReference.Copy(output, input, spec);
            return;
        }
        RequireUIntRange(spec, g, "CopyStrided");
        DispatchStrided("copy_strided" + WordSuffix(g.ElementSize), output, input, g, fresh: g.DestinationCoversOutput);
    }

    /// <inheritdoc />
    public void AccumulateStrided(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        using OpScope _op = EnterOp();
        StridedCopyGeometry g = StridedCopyContract.ValidateAccumulate(output, input, spec);
        if (output.DType == DType.BF16)
        {
            StridedCopyReference.Accumulate(output, input, spec);
            return;
        }
        RequireUIntRange(spec, g, "AccumulateStrided");
        DispatchStrided("accumulate_strided" + DtypeSuffix(output.DType), output, input, g, fresh: false);
    }

    /// <inheritdoc />
    public void Pad(Tensor output, Tensor input, in PadSpec spec)
    {
        using OpScope _op = EnterOp();
        PadGeometry g = PadContract.Validate(output, input, spec);
        if (g.ElementSize == 1)
        {
            StridedCopyReference.Pad(output, input, spec);
            return;
        }
        if (g.ElementCount > uint.MaxValue || input.ElementCount > uint.MaxValue)
            throw new NotSupportedException($"Vulkan Pad addresses at most {uint.MaxValue} elements; got output {output.Shape}.");
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(g.ElementCount * g.ElementSize));
        try
        {
            VulkanKernel kernel = GetKernel("pad_strided" + WordSuffix(g.ElementSize), 2, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)g.ElementCount);
            pc.U32((uint)g.Rank);
            pc.U32((uint)g.Mode);
            for (int axis = 0; axis < StridedView.MaxRank; axis++) pc.U32(axis < g.Rank ? (uint)g.OutputDim(axis) : 0u);
            for (int axis = 0; axis < StridedView.MaxRank; axis++) pc.U32(axis < g.Rank ? (uint)g.InputDim(axis) : 0u);
            for (int axis = 0; axis < StridedView.MaxRank; axis++) pc.U32(axis < g.Rank ? (uint)g.Before(axis) : 0u);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(g.ElementCount, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Pad dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <summary>Runs one strided shader over the collapsed geometry; <paramref name="fresh"/> allocates the output instead
    /// of binding its current buffer, which an in-place scatter or accumulate must keep.</summary>
    private void DispatchStrided(string shader, Tensor output, Tensor input, in StridedCopyGeometry g, bool fresh)
    {
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer outBuf = fresh ? _xfer.AllocateDevice((ulong)(output.ElementCount * output.DType.SizeInBytes)) : GetBuffer(output);
        try
        {
            VulkanKernel kernel = GetKernel(shader, 2, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)g.ElementCount);
            pc.U32((uint)g.Rank);
            for (int axis = 0; axis < StridedView.MaxRank; axis++) pc.U32(axis < g.Rank ? (uint)g.Dim(axis) : 0u);
            for (int axis = 0; axis < StridedView.MaxRank; axis++) pc.U32(axis < g.Rank ? (uint)g.SourceStride(axis) : 0u);
            for (int axis = 0; axis < StridedView.MaxRank; axis++) pc.U32(axis < g.Rank ? (uint)g.DestinationStride(axis) : 0u);
            pc.U32((uint)g.SourceOffset);
            pc.U32((uint)g.DestinationOffset);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(g.ElementCount, LocalX1D));
            // In place or fresh, the rebind is what tells the tensor its device contents are current.
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error($"Vulkan {shader} dispatch failed", ex);
            if (fresh) outBuf.Dispose();
            throw;
        }
    }

    /// <summary>The shaders index with 32-bit words: every element the views reach must fit.</summary>
    private static void RequireUIntRange(in StridedCopySpec spec, in StridedCopyGeometry g, string operation)
    {
        if (g.ElementCount > uint.MaxValue || spec.Source.MaxElementIndex > uint.MaxValue || spec.Destination.MaxElementIndex > uint.MaxValue)
            throw new NotSupportedException($"Vulkan {operation} addresses at most {uint.MaxValue} elements; got {spec}.");
    }

    /// <summary>Shader variant for an element width: 16, 32 or 64-bit words.</summary>
    private static string WordSuffix(int elementSize) => elementSize switch
    {
        2 => "_u16",
        4 => "_u32",
        _ => "_u64",
    };
}
