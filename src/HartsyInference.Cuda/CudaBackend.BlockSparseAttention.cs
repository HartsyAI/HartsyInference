using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

// 3D block-sparse attention on the device (Kernels/attention/block_sparse_attention.cu).
public sealed partial class CudaBackend
{
    /// <inheritdoc/>
    public bool HasBlockSparseAttention
    {
        get
        {
            EnsureKernels();
            return _kernels is { HasBlockSparseAttention: true };
        }
    }

    /// <inheritdoc/>
    /// <remarks>Head dim 16, 32, 64 or 128, and at most one key block per 4 bytes of a block's default shared memory (12288 where
    /// that is 48 KB). The block means and scores are the reference's bit for bit, so the selection is the reference's; the attention
    /// is F32 with an online softmax. Nothing waits for the device: the output and the selection report stay on it until read.</remarks>
    public unsafe void BlockSparseAttention(Tensor output, Tensor query, Tensor key, Tensor value, BlockSparseAttentionOptions options,
        Tensor? selected = null)
    {
        BlockSparseAttentionContract.Validate(output, query, key, value, options, selected);
        using OpScope _op = EnterOp(nameof(BlockSparseAttention));
        EnsureKernels();
        if (_kernels is not { HasBlockSparseAttention: true })
            throw new AttentionUnsupportedException("BlockSparseAttention is not available: block_sparse_attention.ptx is not loaded on this device.");
        int batch = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[3];
        int queryTokens = (int)query.Shape[2], keyTokens = (int)key.Shape[2];
        int queryBlocks = options.QueryBlocks, keyBlocks = options.KeyBlocks, chosen = options.SelectedBlocks;
        if (!_kernels.SupportsBlockSparseHeadDim(dim))
            throw new AttentionUnsupportedException($"CUDA BlockSparseAttention takes a head dim of 16, 32, 64 or 128 that the card has the shared memory for; got {dim}.");
        if (keyBlocks > _kernels.MaxBlockSparseKeyBlocks)
            throw new AttentionUnsupportedException($"CUDA BlockSparseAttention takes up to {_kernels.MaxBlockSparseKeyBlocks} key blocks; got {keyBlocks}.");
        if (!CudaKernels.GridYzFits(heads, batch))
            throw new AttentionUnsupportedException($"CUDA BlockSparseAttention takes a grid of heads x batch; got {query.Shape}.");

        nint stream = _stream.Handle;
        nuint outBytes = GpuTransferHelper.ByteSize(output);
        ulong pOut = 0, pQ = 0, pK = 0, pV = 0, pQueryMeans = 0, pKeyMeans = 0, pSelected = 0;
        bool cachedOutput = false, cachedSelected = false;
        try
        {
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            // No selected block: the output is zero and the report holds nothing.
            if (chosen == 0)
            {
                CudaDriverApi.cuMemsetD8Async(pOut, 0, outBytes, stream).ThrowOnError();
                GpuTransferHelper.CacheActivation(output, pOut, outBytes);
                cachedOutput = true;
                return;
            }
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            long sequences = (long)batch * heads;
            pQueryMeans = GpuTransferHelper.AllocateDevice((nuint)(sequences * queryBlocks * dim * sizeof(float)));
            pKeyMeans = GpuTransferHelper.AllocateDevice((nuint)(sequences * keyBlocks * dim * sizeof(float)));
            nuint selectedBytes = (nuint)(sequences * queryBlocks * chosen * sizeof(int));
            pSelected = GpuTransferHelper.AllocateDevice(selectedBytes);

            DType type = query.DType;
            _kernels.LaunchBlockSparseMeans(pQueryMeans, pQ, type, batch, heads, queryTokens, dim, queryBlocks, options.QueryGrid,
                options.QueryBlock, stream);
            _kernels.LaunchBlockSparseMeans(pKeyMeans, pK, type, batch, heads, keyTokens, dim, keyBlocks, options.KeyGrid,
                options.KeyBlock, stream);
            _kernels.LaunchBlockSparseSelect(pSelected, pQueryMeans, pKeyMeans, batch, heads, dim, queryBlocks, keyBlocks, chosen, stream);
            _kernels.LaunchBlockSparseAttend(pOut, pQ, pK, pV, pSelected, type, output.DType == DType.BF16, batch, heads, dim,
                queryTokens, keyTokens, options, options.Scale ?? 1f / MathF.Sqrt(dim), stream);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            if (selected is not null)
            {
                GpuTransferHelper.CacheActivation(selected, pSelected, selectedBytes);
                cachedSelected = true;
            }
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (!cachedSelected) GpuTransferHelper.FreeDevice(pSelected);
            foreach (ulong pointer in (ReadOnlySpan<ulong>)[pQ, pK, pV, pQueryMeans, pKeyMeans])
                GpuTransferHelper.FreeDevice(pointer);
        }
    }
}
