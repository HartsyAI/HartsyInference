using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The Mamba-1 selective scan (IBackend.SelectiveScan) on the device. Validates through SelectiveScanReference, and the shader
// holds every state count up to the reference's limit, so there is no fallback.
public sealed partial class VulkanBackend
{
    /// <summary>Workgroup size of ssm_selective_scan: one workgroup per (row, channel).</summary>
    private const uint SelectiveScanLocal = 32;

    /// <inheritdoc/>
    public void SelectiveScan(Tensor output, Tensor u, Tensor delta, Tensor a, Tensor b, Tensor c, Tensor d, Tensor? z, Tensor? state,
        int rows, int frames, int channels, int states)
    {
        using OpScope _op = EnterOp();
        SelectiveScanReference.Validate(output, u, delta, a, b, c, d, z, state, rows, frames, channels, states);
        VulkanBuffer uBuf = GetBuffer(u);
        VulkanBuffer deltaBuf = GetBuffer(delta);
        VulkanBuffer aBuf = GetBuffer(a);
        VulkanBuffer bBuf = GetBuffer(b);
        VulkanBuffer cBuf = GetBuffer(c);
        VulkanBuffer dBuf = GetBuffer(d);
        // Absent z and state bind u as a stand-in: the push constants tell the shader not to read or write them.
        VulkanBuffer zBuf = z is null ? uBuf : GetBuffer(z);
        VulkanBuffer stateBuf = state is null ? uBuf : GetBuffer(state);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel scan = GetKernel("ssm_selective_scan", storageBufferCount: 9, RowSpec(SelectiveScanLocal));
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)rows);
            pc.U32((uint)frames);
            pc.U32((uint)channels);
            pc.U32((uint)states);
            pc.U32(z is null ? 0u : 1u);
            pc.U32(state is null ? 0u : 1u);
            Span<ulong> scanBufs = stackalloc ulong[]
            {
                outBuf.Handle, uBuf.Handle, deltaBuf.Handle, aBuf.Handle, bBuf.Handle, cBuf.Handle, dBuf.Handle, zBuf.Handle,
                stateBuf.Handle,
            };
            Dispatch(scan, scanBufs, pc.Written, (uint)channels, (uint)rows);

            CacheOutput(output, outBuf);
            if (state is not null)
            {
                // The carried state was advanced in place on its own buffer: re-cache it.
                CacheOutput(state, stateBuf);
            }
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SelectiveScan dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }
}
