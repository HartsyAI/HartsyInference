using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// The speculative-decode merge on the device (Kernels/lm/flash_attn_decode_merge.cu).
public sealed partial class CudaBackend
{
    /// <inheritdoc/>
    public bool HasDecodeMerge
    {
        get
        {
            EnsureKernels();
            return _kernels is { HasDecodeMerge: true };
        }
    }

    /// <inheritdoc/>
    /// <remarks>The operands stay on the device: the prefix output and its LSE are read where a decode left them, so a
    /// decode followed by a merge needs no host round trip. The mask is the host's and is uploaded for the call.</remarks>
    public unsafe void DecodeTreeMerge(Tensor merged, Tensor prefixOutput, Tensor prefixLse, Tensor query, Tensor key, Tensor value,
        ReadOnlySpan<int> mask, float? scale = null)
    {
        DecodeMergeReference.CheckOperands(merged, prefixOutput, prefixLse, query, key, value, mask, scale);
        int batch = (int)query.Shape[0], heads = (int)query.Shape[1], rows = (int)query.Shape[2], dim = (int)query.Shape[3];
        int kvHeads = (int)key.Shape[1];
        using OpScope _op = EnterOp(nameof(DecodeTreeMerge));
        EnsureKernels();
        if (_kernels is not { HasDecodeMerge: true })
            throw new NotSupportedException("DecodeTreeMerge is not available: flash_attn_decode_merge.ptx is not in the Ptx folder.");
        if ((long)batch * rows * heads > int.MaxValue)
            throw new NotSupportedException($"DecodeTreeMerge launches one block per output row; {(long)batch * rows * heads} rows is past the grid.");
        using Tensor maskTensor = new(new TensorShape(rows), DType.I32);
        mask.CopyTo(maskTensor.AsSpan<int>());
        ulong pPrefix = 0, pLse = 0, pQ = 0, pK = 0, pV = 0, pMask = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pPrefix = GpuTransferHelper.CopyToDevice(prefixOutput);
            pLse = GpuTransferHelper.CopyToDevice(prefixLse);
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            pMask = GpuTransferHelper.CopyToDevice(maskTensor);
            AttentionEvidence.Observed(AttentionEvidenceKind.StagingCopy, "cuda.flash.decode.merge.mask", rows * sizeof(int));
            nuint outBytes = GpuTransferHelper.ByteSize(merged);
            // Always a fresh buffer: when the merged output is the prefix output, binding it below replaces that tensor's
            // device copy, so nothing is read and written through the same pointer.
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels.LaunchFlashDecodeMerge(pOut, pPrefix, pLse, pQ, pK, pV, pMask, batch, rows, heads, kvHeads, dim,
                scale ?? 1f / MathF.Sqrt(dim), _stream.Handle);
            GpuTransferHelper.CacheActivation(merged, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pPrefix);
            GpuTransferHelper.FreeDevice(pLse);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pMask);
        }
    }
}
