using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Device row gather for a device-resident input: output[m] = input[rowIndices[m]] as one buffer copy per row. The
// prefill's last-row slice uses it, so the hidden state is never read back to the host just to pick one row.
public sealed partial class VulkanBackend
{
    /// <inheritdoc/>
    /// <remarks>Returns false, changing nothing, unless the input is already on the device and both tensors are F32 with
    /// in-range rows. The caller then falls back to <see cref="HartsyInference.Core.Backends.IBackend.GatherRows"/>, which
    /// reads the input on the host and reports bad indices. Not taken while a step graph is captured.</remarks>
    public unsafe bool TryGatherRowsResident(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        if (_capturingStepGraph || output.DType != DType.F32 || input.DType != DType.F32 || input.Shape.Rank < 1)
            return false;
        long width = input.Shape[input.Shape.Rank - 1];
        if (width <= 0 || width > int.MaxValue || output.ElementCount != (long)rowIndices.Length * width)
            return false;
        long inputRows = input.ElementCount / width;
        for (int i = 0; i < rowIndices.Length; i++)
        {
            if (rowIndices[i] < 0 || rowIndices[i] >= inputRows) return false;
        }
        if (!_xfer.TryGetCached(input, out VulkanBuffer? src) || src is null) return false;

        using OpScope _op = EnterOp();
        ulong rowBytes = (ulong)width * sizeof(float);
        VulkanBuffer outBuf = _xfer.AllocateDevice(Math.Max((ulong)output.ElementCount * sizeof(float), 4UL));
        try
        {
            if (rowIndices.Length > 0)
            {
                VkBufferCopy[] regions = new VkBufferCopy[rowIndices.Length];
                for (int m = 0; m < rowIndices.Length; m++)
                {
                    regions[m] = new VkBufferCopy
                    {
                        srcOffset = (ulong)rowIndices[m] * rowBytes,
                        dstOffset = (ulong)m * rowBytes,
                        size = rowBytes,
                    };
                }
                nint cb = _stream.AcquireRecording();
                VulkanCommandStream.RecordComputeToCopyBarrierOn(cb);
                fixed (VkBufferCopy* p = regions)
                    VulkanApi.vkCmdCopyBuffer(cb, src.Handle, outBuf.Handle, (uint)regions.Length, (nint)p);
                VulkanCommandStream.RecordCopyToComputeBarrierOn(cb);
            }
            CacheOutput(output, outBuf);
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
        return true;
    }
}
