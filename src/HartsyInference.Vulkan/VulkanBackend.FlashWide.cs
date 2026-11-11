using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The wide flash row (Shaders/sdpa_flash_wide.comp.glsl): unmasked F32 FlashAttention calls with sinks, which sdpa_flash lacks,
// at any head width up to 1024 and with an optional soft-cap, such as the HCA and DSA calls AttentionFlashLowering stages.
public sealed partial class VulkanBackend
{
    /// <summary>Widest head <c>sdpa_flash_wide</c> holds; must equal its <c>MAX_D</c>.</summary>
    private const int FlashWideMaxHeadDim = 1024;

    private bool? _flashWideBuilt;

    /// <summary>Whether <c>sdpa_flash_wide</c> computes <paramref name="c"/>: F32, unmasked, host positions, sinks, no ALiBi,
    /// equal query/key and value widths up to <see cref="FlashWideMaxHeadDim"/>, and a grid within the device limits.</summary>
    /// <remarks>Sinks are required so the row takes only the sink calls the host reference used to compute (the HCA and DSA
    /// layers through AttentionFlashLowering); every sink-less call keeps the route it had.</remarks>
    private bool FlashWideAdmits(in AttentionCall c) =>
        c.Logits.Sinks is not null && !c.Mask.Causal && c.Mask.LegacySlidingWindow == 0 && c.Mask.Bias is null
        && c.Logits.AlibiSlopes is null && !c.Positions.OnDevice
        && c.Query.DType == DType.F32 && c.Key.DType == DType.F32 && c.Value.DType == DType.F32 && c.Output.DType == DType.F32
        && c.QkHeadDim <= FlashWideMaxHeadDim && c.ValueHeadDim == c.QkHeadDim
        && c.Positions.KvLength >= 1 && c.Positions.KvLength <= c.KeyTokens
        && (uint)c.QueryTokens <= Vk.MaxComputeWorkGroupCountX && (uint)c.Heads <= Vk.MaxComputeWorkGroupCountY
        && (uint)c.Batch <= Vk.MaxComputeWorkGroupCountZ
        && (_flashWideBuilt ??= File.Exists(Path.Combine(_spvDir, "sdpa_flash_wide.spv")));

    // One workgroup per (query row, head, batch) over head-major operands; the output is a fresh buffer.
    private void DispatchFlashWide(in AttentionCall c)
    {
        Tensor q = c.Query.Storage, k = c.Key.Storage, v = c.Value.Storage, o = c.Output.Storage;
        Tensor? sink = c.Logits.Sinks;
        int batch = c.Batch, hq = c.Heads, sq = c.QueryTokens, d = c.QkHeadDim;
        List<VulkanBuffer> temps = new();
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)((long)batch * hq * sq * d * sizeof(float)));
        try
        {
            VulkanBuffer sinkBuf = sink is null ? Placeholder(temps) : GetBuffer(sink);
            VulkanKernel kernel = GetKernel("sdpa_flash_wide", storageBufferCount: 5, ReadOnlySpan<SpecConstant>.Empty);
            Span<ulong> bufs = stackalloc ulong[] { GetBuffer(q).Handle, GetBuffer(k).Handle, GetBuffer(v).Handle, sinkBuf.Handle, outBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)hq);
            pc.U32((uint)c.KvHeads);
            pc.U32((uint)sq);
            pc.U32((uint)c.Positions.KvLength);
            pc.U32((uint)c.KeyTokens);
            pc.U32((uint)d);
            pc.F32(c.Scale);
            pc.F32(c.Logits.Softcap);
            pc.U32(sink is null ? 0u : 1u);
            Dispatch(kernel, bufs, pc.Written, (uint)sq, (uint)hq, (uint)batch);
            CacheOutput(o, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan sdpa_flash_wide dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            FreeTemps(temps);
        }
    }
}
