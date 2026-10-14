using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Flash-decoding for one query row (the decode step). Pass 1 (sdpa_flash_decode_split) gives each (split, KV head,
// batch) workgroup the partial softmax state of every query head in that KV group over one fixed run of keys; pass 2
// (sdpa_flash_decode_combine) merges the live splits per query head. Both passes derive their geometry from the same
// cache capacity, so eager and graph decode run identical arithmetic and only the source of kvLen/qOffset differs.
public sealed partial class VulkanBackend
{
    /// <summary>Routes single-row attention through the flash-decoding pair; see <see cref="EngineKnobs.VkFlashDecodeSplit"/>.</summary>
    public bool EnableFlashDecodeSplit { get; set; } = EngineKnobs.VkFlashDecodeSplit.Value;

    /// <summary>Workgroups the split grid aims for per compute unit: one resident wave of subgroup-sized workgroups.</summary>
    private const uint FlashDecodeWorkgroupsPerUnit = 16;

    /// <summary>Compute units assumed when the device does not report any.</summary>
    private const uint FlashDecodeFallbackUnits = 16;

    /// <summary>Upper bound on splits per head: the combine pass reads each live partial once per row.</summary>
    internal const int FlashDecodeMaxSplits = 128;

    /// <summary>Smallest run of keys a split takes; shorter runs spend more on the split's reduction and launch than on keys.</summary>
    internal const int FlashDecodeMinChunk = 16;

    /// <summary>Work cap per split. The split count can force a longer run only when the cache holds more than
    /// <see cref="FlashDecodeMaxSplits"/> times this many keys.</summary>
    internal const int FlashDecodeMaxChunk = 256;

    /// <summary>Query heads per KV head the split kernel keeps in registers (its GROUP specialization bound).</summary>
    private const uint FlashDecodeMaxGroup = 8;

    /// <summary>Head dims per lane the split kernel keeps in registers (its DPL specialization bound).</summary>
    private const uint FlashDecodeMaxDpl = 4;

    /// <summary>Chunk (keys per split) and split count for a KV cache of <paramref name="capacity"/> keys. A pure function
    /// of the capacity, the KV-head count, the batch and the compute units, so every step of one cache gets the same split
    /// grid. The split count aims for <see cref="FlashDecodeWorkgroupsPerUnit"/> workgroups per compute unit; a split's
    /// work is capped at <see cref="FlashDecodeMaxChunk"/> keys unless <see cref="FlashDecodeMaxSplits"/> forces more.</summary>
    internal static (int Chunk, int Splits) FlashDecodeGeometry(int capacity, int kvHeads, int batch, uint computeUnits)
    {
        long units = computeUnits > 0 ? computeUnits : FlashDecodeFallbackUnits;
        long workgroupsPerHead = Math.Max(1L, (long)kvHeads * batch);
        long splitTarget = Math.Clamp((FlashDecodeWorkgroupsPerUnit * units + workgroupsPerHead - 1) / workgroupsPerHead,
            1L, FlashDecodeMaxSplits);
        long chunk = Math.Clamp((capacity + splitTarget - 1) / splitTarget, FlashDecodeMinChunk, FlashDecodeMaxChunk);
        chunk = Math.Max(chunk, (capacity + FlashDecodeMaxSplits - 1) / FlashDecodeMaxSplits);
        int splits = (int)((capacity + chunk - 1) / chunk);
        return ((int)chunk, splits);
    }

    /// <summary>Whether a single-row attention call takes the flash-decoding pair: query length 1, F32 Q/K/V/O, no mask
    /// (callers pass none), consistent batch and head counts, a GQA group of at most <see cref="FlashDecodeMaxGroup"/>,
    /// and a head dim the subgroup divides into at most <see cref="FlashDecodeMaxDpl"/> dims per lane.</summary>
    private bool CanUseFlashDecode(Tensor output, Tensor query, Tensor key, Tensor value, int kvGroup)
    {
        if (!EnableFlashDecodeSplit || query.Shape.Rank != 4 || key.Shape.Rank != 4 || value.Shape.Rank != 4)
            return false;
        if (query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32 || output.DType != DType.F32)
            return false;
        long hq = query.Shape[1], batch = query.Shape[0], headDim = query.Shape[3];
        // kvGroup is the GQA ratio (query heads per KV head), so it is the workgroup's head count, not the KV-head count.
        if (query.Shape[2] != 1 || kvGroup < 1 || kvGroup > FlashDecodeMaxGroup || hq % kvGroup != 0)
            return false;
        long hkv = hq / kvGroup;
        if (key.Shape[0] != batch || value.Shape[0] != batch || key.Shape[1] != hkv || value.Shape[1] != hkv
            || key.Shape[3] != headDim || value.Shape[3] != headDim || key.Shape[2] != value.Shape[2] || key.Shape[2] < 1)
            return false;
        uint lanes = Vk.SubgroupSize;
        if (lanes == 0 || (Vk.SubgroupOps & VkSubgroupFeatureFlags.Arithmetic) == 0)
            return false;
        return headDim % lanes == 0 && headDim / lanes >= 1 && headDim / lanes <= FlashDecodeMaxDpl;
    }

    /// <summary>Flash-decoding over the first <paramref name="kvLen"/> keys (eager, <paramref name="devicePos"/> 0) or over the
    /// device position buffer (graph decode, <paramref name="devicePos"/> set; <paramref name="kvLen"/> and
    /// <paramref name="qOffset"/> are then ignored). The split grid comes from the cache capacity, so it is fixed per cache.</summary>
    private void DispatchFlashDecode(Tensor output, Tensor query, Tensor key, Tensor value, float scale, bool causal,
        int qOffset, int slidingWindow, int kvLen, int kvGroup, ulong devicePos)
    {
        int batch = (int)query.Shape[0], hq = (int)query.Shape[1], headDim = (int)query.Shape[3];
        int hkv = hq / kvGroup, capacity = (int)key.Shape[2];
        bool device = devicePos != 0;
        (int chunk, int splits) = FlashDecodeGeometry(capacity, hkv, batch, Vk.ComputeUnits);

        VulkanBuffer qBuf = GetBuffer(query);
        VulkanBuffer kBuf = GetBuffer(key);
        VulkanBuffer vBuf = GetBuffer(value);
        VulkanBuffer? posBuf = device ? _scalarBuffers[devicePos] : null;

        // Partial state per (batch, query head, split): [m, l, acc[0..headDim)].
        ulong floatBytes = sizeof(float);
        ulong partBytes = (ulong)batch * (ulong)hq * (ulong)splits * (ulong)(headDim + 2) * floatBytes;
        VulkanBuffer partBuf = _xfer.AllocateDevice(partBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)batch * (ulong)hq * (ulong)headDim * floatBytes);
        try
        {
            uint lanes = Vk.SubgroupSize;
            uint dpl = (uint)(headDim / lanes);
            ReadOnlySpan<SpecConstant> splitSpec =
            [
                SpecConstant.UInt(0, lanes),          // workgroup width = subgroup size
                SpecConstant.UInt(1, (uint)kvGroup),  // query heads per KV head
                SpecConstant.UInt(2, dpl),            // head dims per lane
            ];
            ReadOnlySpan<SpecConstant> combineSpec =
            [
                SpecConstant.UInt(0, lanes),
                SpecConstant.UInt(2, dpl),
            ];
            VulkanKernel split = device
                ? GetKernel("sdpa_flash_decode_split_dev", storageBufferCount: 5, splitSpec)
                : GetKernel("sdpa_flash_decode_split", storageBufferCount: 4, splitSpec);
            VulkanKernel combine = device
                ? GetKernel("sdpa_flash_decode_combine_dev", storageBufferCount: 3, combineSpec)
                : GetKernel("sdpa_flash_decode_combine", storageBufferCount: 2, combineSpec);

            Span<byte> pc = stackalloc byte[11 * 4];
            BinaryWriteUInt(pc, 0, (uint)hq);
            BinaryWriteUInt(pc, 4, (uint)hkv);
            BinaryWriteUInt(pc, 8, (uint)capacity);
            BinaryWriteUInt(pc, 12, device ? 0u : (uint)kvLen);
            BinaryWriteUInt(pc, 16, device ? 0u : (uint)qOffset);
            BinaryWriteUInt(pc, 20, (uint)headDim);
            BinaryWriteFloat(pc, 24, scale);
            BinaryWriteUInt(pc, 28, causal ? 1u : 0u);
            BinaryWriteUInt(pc, 32, (uint)slidingWindow);
            BinaryWriteUInt(pc, 36, (uint)splits);
            BinaryWriteUInt(pc, 40, (uint)chunk);

            // Pass 1: one workgroup per (split, KV head, batch); it covers every query head of the KV group.
            Span<ulong> splitBufs = device
                ? stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, posBuf!.Handle, partBuf.Handle }
                : stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, partBuf.Handle };
            Dispatch(split, splitBufs, pc, (uint)splits, (uint)hkv, (uint)batch);

            // Pass 2: one workgroup per (query head, batch); it reads the live splits only.
            if (device)
            {
                Span<ulong> combineBufs = stackalloc ulong[] { partBuf.Handle, posBuf!.Handle, outBuf.Handle };
                Dispatch(combine, combineBufs, pc, (uint)hq, (uint)batch);
            }
            else
            {
                Span<ulong> combineBufs = stackalloc ulong[] { partBuf.Handle, outBuf.Handle };
                Dispatch(combine, combineBufs, pc, (uint)hq, (uint)batch);
            }

            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan flash-decode dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            _xfer.FreeDevice(partBuf);
        }
    }
}
