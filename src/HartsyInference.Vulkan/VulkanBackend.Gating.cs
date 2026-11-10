using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The gated linear unit and the group gate: one shader each, the gate function and the gate half or accumulate as spec
// constants, F32 and F16 activations; BF16 takes the host reference. The fused linear runs the interface's composition.
public sealed partial class VulkanBackend
{
    /// <summary>Spec constant ids of the gating shaders: the activation or gate function, and the gate half or the accumulate flag.</summary>
    private const uint GatingFunctionId = 3, GatingModeId = 4;

    /// <inheritdoc />
    public void Glu(Tensor output, Tensor input, in GluSpec spec)
    {
        using OpScope _op = EnterOp();
        (int inner, long rows) = GluContract.Validate(output, input, spec);
        if (rows == 0) return;
        if (input.DType == DType.BF16)
        {
            GluReference.Apply(output, input, spec);
            return;
        }
        long total = rows * inner;
        if (total > uint.MaxValue)
            throw new NotSupportedException($"Vulkan Glu addresses at most {uint.MaxValue} elements; got {input.Shape}.");
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * output.DType.SizeInBytes));
        try
        {
            SpecConstant[] constants = [SpecConstant.UInt(0, LocalX1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1),
                SpecConstant.UInt(GatingFunctionId, (uint)spec.Activation), SpecConstant.Bool(GatingModeId, spec.Gate == GluGateHalf.First)];
            VulkanKernel kernel = GetKernel("glu" + DtypeSuffix(input.DType), 2, constants);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)inner);
            pc.U32((uint)total);
            pc.F32(spec.SituBeta);
            pc.F32(spec.SituLinearBeta);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, outBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(total, LocalX1D));
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Glu dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public void GroupScale(Tensor output, Tensor input, Tensor gates, in GroupScaleSpec spec)
    {
        using OpScope _op = EnterOp();
        GroupScaleGeometry g = GroupScaleContract.Validate(output, input, gates, spec);
        if (g.Rows == 0) return;
        if (input.DType == DType.BF16 || gates.DType == DType.BF16)
        {
            GroupScaleReference.Apply(output, input, gates, spec);
            return;
        }
        long total = g.Rows * g.Width;
        if (total > uint.MaxValue || spec.GateOffset + (g.Rows - 1) * spec.GateStride + g.Groups > uint.MaxValue)
            throw new NotSupportedException($"Vulkan GroupScale addresses at most {uint.MaxValue} elements; got {input.Shape}.");
        bool inPlace = ReferenceEquals(output, input);
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer gateBuf = GetBuffer(gates);
        // In place the input's buffer is the output's; an accumulate reads the output's current contents.
        VulkanBuffer outBuf = inPlace ? inBuf : spec.Accumulate ? GetBuffer(output) : _xfer.AllocateDevice((ulong)(total * output.DType.SizeInBytes));
        bool owned = !inPlace && !spec.Accumulate;
        try
        {
            string suffix = gates.DType == DType.F16 ? "_f16_f16" : DtypeSuffix(input.DType);
            SpecConstant[] constants = [SpecConstant.UInt(0, LocalX1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1),
                SpecConstant.UInt(GatingFunctionId, (uint)spec.Function), SpecConstant.Bool(GatingModeId, spec.Accumulate)];
            VulkanKernel kernel = GetKernel("group_scale" + suffix, 3, constants);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)total);
            pc.U32((uint)g.Width);
            pc.U32((uint)g.GroupWidth);
            pc.U32((uint)spec.GateStride);
            pc.U32((uint)spec.GateOffset);
            Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, gateBuf.Handle, outBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(total, LocalX1D));
            // Written in place or accumulated: the rebind is what tells the tensor its device contents are current.
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan GroupScale dispatch failed", ex);
            if (owned) outBuf.Dispose();
            throw;
        }
    }
}
