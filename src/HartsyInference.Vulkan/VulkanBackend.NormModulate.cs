using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The generic norm-and-modulate and chunk-activate ops: one row-per-workgroup shader for a norm along the last axis, one
// invocation-per-element shader for the affine and modulation alone, one invocation-per-row shader along the channel axis
// (F32), the spec constants picking the norm, the scale mode and the vectors read. BF16 activations take the host reference.
public sealed partial class VulkanBackend
{
    /// <summary>Spec constant ids of the norm shaders: the norm kind, the scale mode and which vectors are read.</summary>
    private const uint NormKindId = 3, NormOnePlusId = 4, NormWeightId = 5, NormBiasId = 6, NormScaleId = 7, NormShiftId = 8, NormIndexId = 9;

    /// <inheritdoc />
    public void NormModulate(in NormModulateCall call)
    {
        using OpScope _op = EnterOp();
        NormModulateGeometry g = NormModulateContract.Validate(call);
        if (g.Rows == 0) return;
        bool channelKernel = call.Kind != NormKind.None && call.Axis == NormAxis.Channel;
        if (call.Input.DType == DType.BF16 || call.Output.DType == DType.BF16 || (channelKernel && (call.Input.DType != DType.F32 || call.Output.DType != DType.F32)))
        {
            NormModulateReference.Apply(call);
            return;
        }
        long total = g.Rows * g.Width;
        if (total > uint.MaxValue || call.Modulation.ScaleOffset > uint.MaxValue || call.Modulation.ShiftOffset > uint.MaxValue)
            throw new NotSupportedException($"Vulkan NormModulate addresses at most {uint.MaxValue} elements; got {call.Input.Shape}.");
        ModulationSpec mod = call.Modulation;
        VulkanBuffer inBuf = GetBuffer(call.Input);
        // Every binding is filled: an unread vector binds the input as a placeholder and its spec constant turns the read off.
        VulkanBuffer wBuf = call.Weight is null ? inBuf : GetBuffer(call.Weight);
        VulkanBuffer bBuf = call.Bias is null ? inBuf : GetBuffer(call.Bias);
        VulkanBuffer scBuf = mod.Scale is null ? inBuf : GetBuffer(mod.Scale);
        VulkanBuffer shBuf = mod.Shift is null ? inBuf : GetBuffer(mod.Shift);
        VulkanBuffer ixBuf = mod.RowIndex is null ? inBuf : GetBuffer(mod.RowIndex);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(call.Output.ElementCount * call.Output.DType.SizeInBytes));
        try
        {
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            if (channelKernel)
            {
                SpecConstant[] spec = [SpecConstant.UInt(0, LocalX1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1),
                    SpecConstant.UInt(NormKindId, (uint)call.Kind), SpecConstant.Bool(NormWeightId, call.Weight is not null), SpecConstant.Bool(NormBiasId, call.Bias is not null)];
                VulkanKernel kernel = GetKernel("norm_modulate_channel_f32", 4, spec);
                pc.U32((uint)g.Rows);
                pc.U32((uint)g.Width);
                pc.U32((uint)g.Spatial);
                pc.F32(call.Eps);
                Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wBuf.Handle, bBuf.Handle, outBuf.Handle };
                Dispatch(kernel, bufs, pc.Written, GroupCount(g.Rows, LocalX1D));
            }
            else
            {
                bool rows = call.Kind != NormKind.None;
                string suffix = call.Input.DType == call.Output.DType ? DtypeSuffix(call.Input.DType) : DtypeSuffix(call.Input.DType) + DtypeSuffix(call.Output.DType);
                if (!rows && call.Input.DType != call.Output.DType)
                    throw new NotSupportedException($"Vulkan NormModulate without a norm keeps one activation dtype; got input {call.Input.DType}, output {call.Output.DType}.");
                SpecConstant[] spec = [SpecConstant.UInt(0, rows ? NormRowLocal : LocalX1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1),
                    SpecConstant.UInt(NormKindId, (uint)call.Kind), SpecConstant.Bool(NormOnePlusId, mod.Mode == ScaleMode.OnePlus),
                    SpecConstant.Bool(NormWeightId, call.Weight is not null), SpecConstant.Bool(NormBiasId, call.Bias is not null),
                    SpecConstant.Bool(NormScaleId, mod.Scale is not null), SpecConstant.Bool(NormShiftId, mod.Shift is not null), SpecConstant.Bool(NormIndexId, mod.RowIndex is not null)];
                VulkanKernel kernel = GetKernel((rows ? "norm_modulate_rows" : "norm_modulate_flat") + suffix, 7, spec);
                if (rows)
                {
                    pc.U32((uint)g.Rows);
                    pc.U32((uint)g.Width);
                }
                else
                {
                    pc.U32((uint)total);
                    pc.U32((uint)g.Width);
                    pc.U32((uint)g.Spatial);
                }
                pc.U32((uint)g.RowsPerMod);
                pc.U32((uint)g.RowStride);
                pc.U32((uint)g.ScaleOffset);
                pc.U32((uint)g.ShiftOffset);
                if (rows) pc.F32(call.Eps);
                Span<ulong> bufs = stackalloc ulong[] { inBuf.Handle, wBuf.Handle, bBuf.Handle, scBuf.Handle, shBuf.Handle, ixBuf.Handle, outBuf.Handle };
                Dispatch(kernel, bufs, pc.Written, rows ? (uint)g.Rows : GroupCount(total, LocalX1D));
            }
            CacheOutput(call.Output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan NormModulate dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public void ChunkActivate(ReadOnlySpan<Tensor> outputs, Tensor input, ReadOnlySpan<ChunkActivation> activations)
    {
        using OpScope _op = EnterOp();
        (long rows, int width) = ChunkActivateReference.Validate(outputs, input, activations);
        long total = rows * width * outputs.Length;
        if (total > uint.MaxValue)
            throw new NotSupportedException($"Vulkan ChunkActivate addresses at most {uint.MaxValue} elements; got {input.Shape}.");
        VulkanBuffer inBuf = GetBuffer(input);
        ulong outBytes = (ulong)(rows * width * sizeof(float));
        VulkanBuffer[] outBufs = new VulkanBuffer[outputs.Length];
        for (int c = 0; c < outputs.Length; c++) outBufs[c] = _xfer.AllocateDevice(outBytes);
        try
        {
            VulkanKernel kernel = GetKernel("chunk_activate", 1 + ChunkActivateReference.MaxChunks, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)total);
            pc.U32((uint)width);
            pc.U32((uint)outputs.Length);
            for (int c = 0; c < ChunkActivateReference.MaxChunks; c++) pc.U32(c < activations.Length ? (uint)activations[c] : 0u);
            // The output bindings past the chunk count bind the first output as a placeholder; the shader never writes them.
            Span<ulong> bufs = stackalloc ulong[1 + ChunkActivateReference.MaxChunks];
            bufs[0] = inBuf.Handle;
            for (int c = 0; c < ChunkActivateReference.MaxChunks; c++) bufs[1 + c] = outBufs[Math.Min(c, outputs.Length - 1)].Handle;
            Dispatch(kernel, bufs, pc.Written, GroupCount(total, LocalX1D));
            for (int c = 0; c < outputs.Length; c++) CacheOutput(outputs[c], outBufs[c]);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan ChunkActivate dispatch failed", ex);
            foreach (VulkanBuffer b in outBufs) b.Dispose();
            throw;
        }
    }
}
