using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The speculative-decode merge on the device (Shaders/sdpa_decode_merge.comp.glsl).
public sealed partial class VulkanBackend
{
    /// <inheritdoc/>
    public bool HasDecodeMerge => true;

    /// <inheritdoc/>
    /// <remarks>One workgroup per output row; the operands stay on the device and the mask is uploaded for the call. The
    /// result always goes to a fresh buffer, which replaces the merged tensor's device copy, so merging in place never
    /// reads and writes one buffer.</remarks>
    public void DecodeTreeMerge(Tensor merged, Tensor prefixOutput, Tensor prefixLse, Tensor query, Tensor key, Tensor value,
        ReadOnlySpan<int> mask, float? scale = null)
    {
        DecodeMergeReference.CheckOperands(merged, prefixOutput, prefixLse, query, key, value, mask, scale);
        int batch = (int)query.Shape[0], heads = (int)query.Shape[1], rows = (int)query.Shape[2], dim = (int)query.Shape[3];
        int kvHeads = (int)key.Shape[1];
        if (heads > ushort.MaxValue || batch > ushort.MaxValue)
            throw new NotSupportedException($"DecodeTreeMerge dispatches one workgroup per (head, row, sequence); {heads} heads or {batch} sequences is past the grid.");
        using OpScope _op = EnterOp();
        using Tensor maskTensor = new(new TensorShape(rows), DType.I32);
        mask.CopyTo(maskTensor.AsSpan<int>());
        VulkanBuffer qBuf = GetBuffer(query), kBuf = GetBuffer(key), vBuf = GetBuffer(value);
        VulkanBuffer prefixBuf = GetBuffer(prefixOutput), lseBuf = GetBuffer(prefixLse), maskBuf = GetBuffer(maskTensor);
        AttentionEvidence.Observed(AttentionEvidenceKind.StagingCopy, "vulkan.decode.merge.mask", rows * sizeof(int));
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)merged.ElementCount * sizeof(float));
        try
        {
            VulkanKernel kernel = GetKernel("sdpa_decode_merge", storageBufferCount: 7, default);
            Span<byte> pc = stackalloc byte[5 * 4];
            BinaryWriteUInt(pc, 0, (uint)heads);
            BinaryWriteUInt(pc, 4, (uint)kvHeads);
            BinaryWriteUInt(pc, 8, (uint)rows);
            BinaryWriteUInt(pc, 12, (uint)dim);
            BinaryWriteFloat(pc, 16, scale ?? 1f / MathF.Sqrt(dim));
            Span<ulong> buffers = stackalloc ulong[]
            {
                qBuf.Handle, kBuf.Handle, vBuf.Handle, prefixBuf.Handle, lseBuf.Handle, maskBuf.Handle, outBuf.Handle,
            };
            Dispatch(kernel, buffers, pc, (uint)heads, (uint)rows, (uint)batch);
            CacheOutput(merged, outBuf);
        }
        catch (Exception)
        {
            // Through the cache: the dispatch may already be recorded against the buffer, and the release is deferred past it.
            _xfer.FreeDevice(outBuf);
            throw;
        }
    }
}
