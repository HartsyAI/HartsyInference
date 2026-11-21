using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Flash-decoding for a decode whose sequences have their own key lengths, with the LSE (AttentionCall.SequenceLengths and
// AttentionCall.Lse). The two passes of VulkanBackend.FlashDecode.cs with the length read per sequence and one workgroup
// per query row: sdpa_flash_decode_lengths_split, then sdpa_flash_decode_lengths_combine.
public sealed partial class VulkanBackend
{
    /// <summary>The most partial-state workspace a decode with sequence lengths may ask for; past it the row declines and
    /// the call is refused.</summary>
    private const long FlashDecodeLengthsWorkspaceLimit = 256L << 20;

    /// <summary>The split geometry of a decode with sequence lengths: the single-length decode's rule over the longest
    /// sequence, with one workgroup per query row of each sequence.</summary>
    private (int Chunk, int Splits) FlashDecodeLengthsGeometry(in AttentionCall c)
    {
        int span = 1;
        foreach (int length in c.SequenceLengths!) span = Math.Max(span, Math.Min(length, c.KeyTokens));
        return FlashDecodeGeometry(span, c.KvHeads, c.Batch * c.QueryTokens, Vk.ComputeUnits);
    }

    /// <summary>The lengths row's admission (C.9). It takes only a call with sequence lengths, so the single-length rows keep
    /// every call they had: F32 head-major Q, K and V, the <c>[B, S, Hq × D]</c> output, a head dim and group the split
    /// shader is built for on this device, and a workspace inside <see cref="FlashDecodeLengthsWorkspaceLimit"/>.</summary>
    private bool FlashDecodeLengthsAdmits(in AttentionCall c)
    {
        if (c.SequenceLengths is null || !EnableFlashDecodeSplit || !(Vk.SubgroupSizeControl || Vk.MinSubgroupSize == Vk.MaxSubgroupSize))
            return false;
        uint lanes = Vk.SubgroupSize;
        int group = c.KvGroup, dim = c.QkHeadDim;
        long rows = (long)c.Batch * c.QueryTokens;
        if (lanes == 0 || (Vk.SubgroupOps & VkSubgroupFeatureFlags.Arithmetic) == 0 || dim % lanes != 0 || dim / lanes > FlashDecodeLengthsMaxDpl
            || dim > FlashDecodeLengthsMaxHeadDim
            || group < 1 || group > FlashDecodeMaxGroup || c.Heads != c.KvHeads * group || c.ValueHeadDim != dim
            || c.Query.Layout != AttentionLayout.HeadMajor || c.Key.Layout != AttentionLayout.HeadMajor
            || c.Value.Layout != AttentionLayout.HeadMajor || c.Output.Layout != AttentionLayout.TokenMajor
            || rows > ushort.MaxValue || c.Heads > ushort.MaxValue
            // Keys and values of one type: the split shader widens one cache type a specialization.
            || c.Key.DType != c.Value.DType || FlashDecodeLengthsCacheKind(c.Key.DType) < 0)
            return false;
        return rows * c.Heads * FlashDecodeLengthsGeometry(in c).Splits * (dim + 2) * sizeof(float) <= FlashDecodeLengthsWorkspaceLimit;
    }

    /// <summary>The split shader's CACHE specialization for a key/value type: 0 F32, 1 F16, 2 BF16; -1 for a type it does
    /// not read.</summary>
    private static int FlashDecodeLengthsCacheKind(DType cache) =>
        cache == DType.F32 ? 0 : cache == DType.F16 ? 1 : cache == DType.BF16 ? 2 : -1;

    /// <summary>A decode whose sequences have their own key lengths, with the LSE when the call names one (C.9).</summary>
    private void DispatchFlashDecodeLengths(in AttentionCall c)
    {
        Tensor output = c.Output.Storage, query = c.Query.Storage, key = c.Key.Storage, value = c.Value.Storage;
        Tensor? lse = c.Lse;
        int batch = c.Batch, rows = c.QueryTokens, hq = c.Heads, hkv = c.KvHeads, headDim = c.QkHeadDim, capacity = c.KeyTokens;
        (int chunk, int splits) = FlashDecodeLengthsGeometry(in c);
        using OpScope _op = EnterOp();
        using Tensor lengths = new(new TensorShape(batch), DType.I32);
        c.SequenceLengths.CopyTo(lengths.AsSpan<int>());

        VulkanBuffer qBuf = GetBuffer(query), kBuf = GetBuffer(key), vBuf = GetBuffer(value);
        // The lengths are the host's and change every step: they are uploaded for the call.
        VulkanBuffer lenBuf = GetBuffer(lengths);
        AttentionEvidence.Observed(AttentionEvidenceKind.StagingCopy, "vulkan.flash.decode.lengths", batch * sizeof(int));

        ulong floatBytes = sizeof(float), outputRows = (ulong)batch * (ulong)rows * (ulong)hq;
        ulong partBytes = outputRows * (ulong)splits * (ulong)(headDim + 2) * floatBytes;
        VulkanBuffer partBuf = _xfer.AllocateDevice(partBytes);
        AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, "vulkan.flash.decode.lengths.partials", (long)partBytes);
        VulkanBuffer outBuf = _xfer.AllocateDevice(outputRows * (ulong)headDim * floatBytes);
        // The combine always writes an LSE; when the call names none it goes to a scratch buffer.
        VulkanBuffer lseBuf = _xfer.AllocateDevice(outputRows * floatBytes);
        try
        {
            uint lanes = Vk.SubgroupSize;
            uint dpl = (uint)(headDim / lanes);
            ReadOnlySpan<SpecConstant> splitSpec =
            [
                SpecConstant.UInt(0, lanes),
                SpecConstant.UInt(1, (uint)c.KvGroup),
                SpecConstant.UInt(2, dpl),
                SpecConstant.UInt(3, (uint)FlashDecodeLengthsCacheKind(key.DType)),
            ];
            ReadOnlySpan<SpecConstant> combineSpec =
            [
                SpecConstant.UInt(0, lanes),
                SpecConstant.UInt(2, dpl),
            ];
            VulkanKernel split = GetKernel("sdpa_flash_decode_lengths_split", storageBufferCount: 5, splitSpec);
            VulkanKernel combine = GetKernel("sdpa_flash_decode_lengths_combine", storageBufferCount: 4, combineSpec);

            Span<byte> pc = stackalloc byte[9 * 4];
            BinaryWriteUInt(pc, 0, (uint)hq);
            BinaryWriteUInt(pc, 4, (uint)hkv);
            BinaryWriteUInt(pc, 8, (uint)capacity);
            BinaryWriteUInt(pc, 12, (uint)rows);
            BinaryWriteUInt(pc, 16, (uint)headDim);
            BinaryWriteFloat(pc, 20, c.Scale);
            BinaryWriteUInt(pc, 24, c.Mask.Causal ? 1u : 0u);
            BinaryWriteUInt(pc, 28, (uint)splits);
            BinaryWriteUInt(pc, 32, (uint)chunk);

            // Pass 1: one workgroup per (split, KV head, sequence and row); it covers every query head of the KV group.
            Span<ulong> splitBufs = stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, lenBuf.Handle, partBuf.Handle };
            Dispatch(split, splitBufs, pc, (uint)splits, (uint)hkv, (uint)(batch * rows));
            // Pass 2: one workgroup per (query head, sequence and row); it reads the live splits only.
            Span<ulong> combineBufs = stackalloc ulong[] { partBuf.Handle, lenBuf.Handle, outBuf.Handle, lseBuf.Handle };
            Dispatch(combine, combineBufs, pc, (uint)hq, (uint)(batch * rows));

            CacheOutput(output, outBuf);
            if (lse is not null) CacheOutput(lse, lseBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan flash-decode with sequence lengths failed", ex);
            throw;
        }
        finally
        {
            // FreeDevice leaves a buffer the cache has taken to the cache: the output and a named LSE after a success, and
            // whichever of them was bound before a failure. Every other buffer is released, deferred past the dispatches
            // already recorded against it.
            _xfer.FreeDevice(partBuf);
            _xfer.FreeDevice(outBuf);
            _xfer.FreeDevice(lseBuf);
        }
    }
}
