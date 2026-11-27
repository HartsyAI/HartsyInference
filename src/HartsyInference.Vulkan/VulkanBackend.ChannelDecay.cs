using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;


namespace HartsyInference.Vulkan;

// The per-key-channel delta-rule prefill (IBackend.SsmChannelDecayPrefill) on the device. Validates through its Core
// reference, and falls back to it when the key head is wider than the shader holds.
public sealed partial class VulkanBackend
{
    /// <summary>Widest key head <c>ssm_channel_decay_prefill</c> holds (<see cref="DeltaScanLocal"/> × its MAX_COLS of 4).</summary>
    private const int ChannelDecayMaxKeyDim = 256;

    /// <inheritdoc/>
    public void SsmChannelDecayPrefill(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor logDecay, Tensor betaRaw,
        int hv, int sv, int sk, int repeat)
    {
        using OpScope _op = EnterOp();
        int seq = GatedDeltaNetReference.ValidateChannelDecayScan(output, state, q, k, v, logDecay, betaRaw, hv, sv, sk, repeat);
        if (sk > ChannelDecayMaxKeyDim)
        {
            GatedDeltaNetReference.ChannelDecayScan(output, state, q, k, v, logDecay, betaRaw, hv, sv, sk, repeat);
            return;
        }
        VulkanBuffer stateBuf = GetBuffer(state);
        VulkanBuffer qBuf = GetBuffer(q);
        VulkanBuffer kBuf = GetBuffer(k);
        VulkanBuffer vBuf = GetBuffer(v);
        VulkanBuffer decayBuf = GetBuffer(logDecay);
        VulkanBuffer betaBuf = GetBuffer(betaRaw);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel scan = GetKernel("ssm_channel_decay_prefill", storageBufferCount: 7, RowSpec(DeltaScanLocal));
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)seq);
            pc.U32((uint)hv);
            pc.U32((uint)sv);
            pc.U32((uint)sk);
            pc.U32((uint)repeat);
            Span<ulong> scanBufs = stackalloc ulong[]
            {
                outBuf.Handle, stateBuf.Handle, qBuf.Handle, kBuf.Handle, vBuf.Handle, decayBuf.Handle, betaBuf.Handle,
            };
            Dispatch(scan, scanBufs, pc.Written, (uint)sv, (uint)hv);

            CacheOutput(output, outBuf);
            // The recurrent state was advanced in place on its own buffer: re-cache it.
            CacheOutput(state, stateBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SsmChannelDecayPrefill dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }
}
