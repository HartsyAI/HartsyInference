using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// 3D block-sparse attention on the device (Shaders/block_sparse_*.comp.glsl): block means, block selection, attention.
public sealed partial class VulkanBackend
{
    /// <summary>The widest head dim the attention pass stages in shared memory (MAX_DIM in the shaders).</summary>
    private const int BlockSparseMaxHeadDim = 128;

    /// <summary>The most key blocks the selection ranks in one workgroup (MAX_KEY_BLOCKS in block_sparse_select).</summary>
    private const int BlockSparseMaxKeyBlocks = 7680;

    /// <summary>Query rows one attention workgroup takes (QT in block_sparse_attend).</summary>
    private const int BlockSparseQueryTile = 32;

    /// <summary>The workgroup memory the selection fills.</summary>
    private const uint BlockSparseSharedBytes = 32 * 1024;

    /// <inheritdoc/>
    public bool HasBlockSparseAttention => Vk.MaxComputeSharedMemoryBytes >= BlockSparseSharedBytes;

    /// <inheritdoc/>
    /// <remarks>Head dim up to 128 and up to 7680 key blocks; BF16 takes a device with 16-bit storage. The block means and
    /// scores are the reference's operations, uncontracted and with a correctly rounded divide, so the selection is the
    /// reference's where the driver keeps subnormal values; the attention is F32 with an online softmax. Nothing waits for the
    /// device: the output and the selection report stay on it until read.</remarks>
    public void BlockSparseAttention(Tensor output, Tensor query, Tensor key, Tensor value, BlockSparseAttentionOptions options,
        Tensor? selected = null)
    {
        BlockSparseAttentionContract.Validate(output, query, key, value, options, selected);
        if (!HasBlockSparseAttention)
            throw new AttentionUnsupportedException(
                $"Vulkan BlockSparseAttention needs {BlockSparseSharedBytes} bytes of workgroup memory; the device has {Vk.MaxComputeSharedMemoryBytes}.");
        int batch = (int)query.Shape[0], heads = (int)query.Shape[1], dim = (int)query.Shape[3];
        int queryTokens = (int)query.Shape[2], keyTokens = (int)key.Shape[2];
        int queryBlocks = options.QueryBlocks, keyBlocks = options.KeyBlocks, chosen = options.SelectedBlocks;
        bool narrowIn = query.DType == DType.BF16, narrowOut = output.DType == DType.BF16;
        if ((narrowIn || narrowOut) && !(Vk.Storage16Bit && Vk.ShaderInt16))
            throw new AttentionUnsupportedException("Vulkan BlockSparseAttention reads and writes BF16 only on a device with 16-bit storage and shaderInt16.");
        TokenGrid3D queryGrid = options.QueryGrid, queryBlock = options.QueryBlock, keyGrid = options.KeyGrid, keyBlock = options.KeyBlock;
        long tileGroups = queryBlocks * ((queryBlock.Tokens + BlockSparseQueryTile - 1) / BlockSparseQueryTile);
        if (dim > BlockSparseMaxHeadDim || keyBlocks > BlockSparseMaxKeyBlocks)
            throw new AttentionUnsupportedException(
                $"Vulkan BlockSparseAttention takes a head dim up to {BlockSparseMaxHeadDim} and up to {BlockSparseMaxKeyBlocks} key blocks; got {dim} and {keyBlocks}.");
        if (Math.Max(tileGroups, keyBlocks) > Vk.MaxComputeWorkGroupCountX || heads > Vk.MaxComputeWorkGroupCountY
            || batch > Vk.MaxComputeWorkGroupCountZ)
            throw new AttentionUnsupportedException($"Vulkan BlockSparseAttention cannot dispatch {query.Shape} in these blocks on this device.");

        using OpScope _op = EnterOp();
        VulkanBuffer qBuf = GetBuffer(query), kBuf = GetBuffer(key), vBuf = GetBuffer(value);
        long sequences = (long)batch * heads;
        ulong selectionBytes = (ulong)(sequences * queryBlocks * chosen) * sizeof(int);
        // Allocated inside the try, so one that fails leaves none of the earlier ones behind.
        VulkanBuffer? queryMeans = null, keyMeans = null, selection = null, outBuf = null;
        bool outputCached = false, selectionCached = false;
        try
        {
            outBuf = _xfer.AllocateDevice((ulong)output.DType.ComputeByteCount(output.ElementCount));
            // With no selected block there is no report, and the attention pass reads no entry of what is bound in its place.
            selection = _xfer.AllocateDevice(Math.Max(sizeof(int), selectionBytes));
            Span<byte> pc = stackalloc byte[17 * 4];
            if (chosen > 0)
            {
                queryMeans = _xfer.AllocateDevice((ulong)(sequences * queryBlocks * dim) * sizeof(float));
                keyMeans = _xfer.AllocateDevice((ulong)(sequences * keyBlocks * dim) * sizeof(float));
                string means = narrowIn ? "block_sparse_means_bf16" : "block_sparse_means_f32";
                BlockSparseMeans(means, qBuf, queryMeans, queryTokens, dim, heads, queryBlocks, queryGrid, queryBlock, batch, pc);
                BlockSparseMeans(means, kBuf, keyMeans, keyTokens, dim, heads, keyBlocks, keyGrid, keyBlock, batch, pc);

                BinaryWriteUInt(pc, 0, (uint)dim);
                BinaryWriteUInt(pc, 4, (uint)heads);
                BinaryWriteUInt(pc, 8, (uint)queryBlocks);
                BinaryWriteUInt(pc, 12, (uint)keyBlocks);
                BinaryWriteUInt(pc, 16, (uint)chosen);
                Span<ulong> selectBufs = stackalloc ulong[] { selection.Handle, queryMeans.Handle, keyMeans.Handle };
                Dispatch(GetKernel("block_sparse_select", storageBufferCount: 3, default), selectBufs, pc[..20], (uint)queryBlocks, (uint)heads, (uint)batch);
            }

            ReadOnlySpan<int> fields =
            [
                queryTokens, keyTokens, dim, heads, queryBlocks, chosen, queryGrid.Height, queryGrid.Width, queryBlock.Frames,
                queryBlock.Height, queryBlock.Width, keyGrid.Height, keyGrid.Width, keyBlock.Frames, keyBlock.Height, keyBlock.Width,
            ];
            for (int i = 0; i < fields.Length; i++) BinaryWriteUInt(pc, 4 * i, (uint)fields[i]);
            BinaryWriteFloat(pc, 4 * fields.Length, options.Scale ?? 1f / MathF.Sqrt(dim));
            string attend = (narrowIn, narrowOut) switch
            {
                (false, false) => "block_sparse_attend_f32",
                (false, true) => "block_sparse_attend_f32_to_bf16",
                (true, false) => "block_sparse_attend_bf16_to_f32",
                _ => "block_sparse_attend_bf16",
            };
            Span<ulong> attendBufs = stackalloc ulong[] { qBuf.Handle, kBuf.Handle, vBuf.Handle, selection.Handle, outBuf.Handle };
            Dispatch(GetKernel(attend, storageBufferCount: 5, default), attendBufs, pc, (uint)tileGroups, (uint)heads, (uint)batch);

            CacheOutput(output, outBuf);
            outputCached = true;
            if (selected is not null && chosen > 0)
            {
                CacheOutput(selected, selection);
                selectionCached = true;
            }
        }
        finally
        {
            if (!outputCached && outBuf is not null) _xfer.FreeDevice(outBuf);
            if (!selectionCached && selection is not null) _xfer.FreeDevice(selection);
            if (queryMeans is not null) _xfer.FreeDevice(queryMeans);
            if (keyMeans is not null) _xfer.FreeDevice(keyMeans);
        }
    }

    private void BlockSparseMeans(string shader, VulkanBuffer rows, VulkanBuffer means, int tokens, int dim, int heads, int blocks,
        TokenGrid3D grid, TokenGrid3D box, int batch, Span<byte> pc)
    {
        ReadOnlySpan<int> fields = [tokens, dim, heads, blocks, grid.Height, grid.Width, box.Frames, box.Height, box.Width];
        for (int i = 0; i < fields.Length; i++) BinaryWriteUInt(pc, 4 * i, (uint)fields[i]);
        Span<ulong> buffers = stackalloc ulong[] { rows.Handle, means.Handle };
        Dispatch(GetKernel(shader, storageBufferCount: 2, default), buffers, pc[..(4 * fields.Length)], (uint)blocks, (uint)heads, (uint)batch);
    }
}
