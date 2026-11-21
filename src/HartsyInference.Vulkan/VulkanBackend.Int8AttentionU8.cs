using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// INT8 attention on Vulkan, the attention itself: the recurrence of Int8AttentionV2Reference.Attend for u8
// probabilities, per-key K scales, one key segment and no mask, over the operands the earlier parts quantize
// (Shaders/int8_attn_u8_tile, int8_attn_u8_drain). The reference run with Int8AttentionExp2.Evaluate as its exp2 is
// the specification: the tile maxima, P bytes, sums and the output are its bits, except that a device which reads
// subnormals as zero drops a drain term the reference carries through one (at most about 2^-104; see the drain shader).
public sealed partial class VulkanBackend
{
    /// <summary>The first pass: for each query row and key tile, the scores, the tile's largest score, the u8
    /// probabilities against it and their sum.</summary>
    /// <param name="tileMax">Output, F32 <c>[heads · rows, tiles]</c>; a NaN where a score of the tile is one.</param>
    /// <param name="tileSums">Output, U32 <c>[heads · rows, tiles]</c>.</param>
    /// <param name="probabilities">Output, U8 <c>[heads · rows, tiles · blockKeys]</c>: each tile padded to a whole
    /// tile of key slots; zero where the tile's maximum is not finite.</param>
    /// <param name="queryBytes">I8 <c>[heads · rows, Dp]</c>.</param>
    /// <param name="queryScales">F32, one a query row.</param>
    /// <param name="queryMeanDots">F32, one a query row, or null where the keys were not centered. Only whether each
    /// is finite matters: the first key segment's offset is left out of the scores.</param>
    /// <param name="keyBytes">I8 <c>[kvHeads · keys, Dp]</c>.</param>
    /// <param name="keyScales">F32, one a key.</param>
    /// <param name="heads">Query heads, the batch folded in; a whole multiple of <paramref name="kvHeads"/>.</param>
    /// <param name="kvHeads">KV heads, the batch folded in.</param>
    /// <param name="blockKeys">Keys a tile, 64 or 128.</param>
    /// <param name="scale">The call's scale on the scores.</param>
    public void Int8AttentionU8Tiles(Tensor tileMax, Tensor tileSums, Tensor probabilities, Tensor queryBytes,
        Tensor queryScales, Tensor? queryMeanDots, Tensor keyBytes, Tensor keyScales, int heads, int kvHeads, int blockKeys,
        float scale)
    {
        (int rows, int keys, int dp, int tiles) = Int8U8Geometry(queryBytes, keyBytes, heads, kvHeads, blockKeys);
        long rowsTotal = (long)heads * rows, total = rowsTotal * tiles;
        Int8Require(queryScales, DType.F32, rowsTotal, nameof(queryScales));
        if (queryMeanDots is not null) Int8Require(queryMeanDots, DType.F32, rowsTotal, nameof(queryMeanDots));
        Int8Require(keyScales, DType.F32, (long)kvHeads * keys, nameof(keyScales));
        Int8Require(tileMax, DType.F32, total, nameof(tileMax));
        Int8Require(tileSums, DType.U32, total, nameof(tileSums));
        Int8Require(probabilities, DType.U8, total * blockKeys, nameof(probabilities));
        if (total == 0) return;
        RequireShader("int8_attn_u8_tile");

        using OpScope _op = EnterOp();
        using Tensor? zeroDots = queryMeanDots is null ? new Tensor(new TensorShape(rowsTotal), DType.F32) : null;
        if (zeroDots is not null) Fill(zeroDots, 0f);
        VulkanBuffer q = GetBuffer(queryBytes), k = GetBuffer(keyBytes), qs = GetBuffer(queryScales), ks = GetBuffer(keyScales);
        VulkanBuffer dots = GetBuffer(queryMeanDots ?? zeroDots!);
        VulkanBuffer maxBuffer = _xfer.AllocateDevice((ulong)total * sizeof(float));
        VulkanBuffer sumBuffer = _xfer.AllocateDevice((ulong)total * sizeof(uint));
        VulkanBuffer pBuffer = _xfer.AllocateDevice((ulong)total * (ulong)blockKeys);
        try
        {
            VulkanKernel kernel = GetKernel("int8_attn_u8_tile", storageBufferCount: 8, _default1DSpec);
            Span<byte> pc = stackalloc byte[10 * 4];
            BinaryWriteUInt(pc, 0, (uint)rowsTotal);
            BinaryWriteUInt(pc, 4, (uint)rows);
            BinaryWriteUInt(pc, 8, (uint)(heads / kvHeads));
            BinaryWriteUInt(pc, 12, (uint)keys);
            BinaryWriteUInt(pc, 16, (uint)dp);
            BinaryWriteUInt(pc, 20, (uint)blockKeys);
            BinaryWriteUInt(pc, 24, (uint)tiles);
            BinaryWriteFloat(pc, 28, scale * Int8AttentionV2Reference.Log2E);
            BinaryWriteFloat(pc, 32, Int8AttentionV2Reference.U8Offset);
            Span<ulong> buffers = stackalloc ulong[]
            {
                q.Handle, k.Handle, qs.Handle, ks.Handle, dots.Handle, maxBuffer.Handle, sumBuffer.Handle, pBuffer.Handle,
            };
            for (long done = 0; done < total; done += MaxElementsPerDispatch)
            {
                BinaryWriteUInt(pc, 36, (uint)done);
                Dispatch(kernel, buffers, pc, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(tileMax, maxBuffer);
            CacheOutput(tileSums, sumBuffer);
            CacheOutput(probabilities, pBuffer);
        }
        finally
        {
            // FreeDevice leaves a buffer the cache has taken to the cache.
            _xfer.FreeDevice(maxBuffer);
            _xfer.FreeDevice(sumBuffer);
            _xfer.FreeDevice(pBuffer);
        }
    }

    /// <summary>The second pass: the reference's online recurrence over the key tiles of each query row, from what
    /// <see cref="Int8AttentionU8Tiles"/> wrote, into the attention output.</summary>
    /// <param name="output">Output, F32 <c>[heads, rows, outDim]</c> with <c>outDim</c> at most <c>Dp</c>; its <c>rows</c>
    /// is the row count the tile pass ran with, which the element counts of the three tile tensors hold it to. NaN for a row
    /// with a tile whose maximum is not finite, zero where there are no keys.</param>
    /// <param name="tileMax">F32 <c>[heads · rows, tiles]</c>.</param>
    /// <param name="tileSums">U32 <c>[heads · rows, tiles]</c>.</param>
    /// <param name="probabilities">U8 <c>[heads · rows, tiles · blockKeys]</c>.</param>
    /// <param name="valueBytes">I8 <c>[kvHeads · keys, Dp]</c>.</param>
    /// <param name="valueScales">F32 <c>[kvHeads, tiles, Dp]</c> where the scales are per tile, else
    /// <c>[kvHeads, 1, Dp]</c>.</param>
    /// <param name="heads">Query heads, the batch folded in.</param>
    /// <param name="kvHeads">KV heads, the batch folded in.</param>
    /// <param name="blockKeys">Keys a tile, 64 or 128.</param>
    /// <param name="valueScalesPerTile">One V scale a channel for each tile, applied at its drain.</param>
    public void Int8AttentionU8Drain(Tensor output, Tensor tileMax, Tensor tileSums, Tensor probabilities, Tensor valueBytes,
        Tensor valueScales, int heads, int kvHeads, int blockKeys, bool valueScalesPerTile)
    {
        if (valueBytes.DType != DType.I8 || valueBytes.Shape.Rank != 2)
            throw new ArgumentException($"The value bytes are I8 [kvHeads · keys, Dp]; got {valueBytes.DType} {valueBytes.Shape}.", nameof(valueBytes));
        Int8U8Heads(heads, kvHeads, blockKeys);
        int dp = (int)valueBytes.Shape[1];
        if (dp is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(valueBytes), dp, "The padded head dim is 64, 128 or 256.");
        if (valueBytes.Shape[0] % kvHeads != 0)
            throw new ArgumentException($"{valueBytes.Shape[0]} value rows are not a whole number of keys for {kvHeads} KV heads.", nameof(valueBytes));
        int keys = (int)(valueBytes.Shape[0] / kvHeads), tiles = (keys + blockKeys - 1) / blockKeys;
        if (output.DType != DType.F32 || output.Shape.Rank != 3 || output.Shape[0] != heads || output.Shape[2] > dp)
            throw new ArgumentException($"The output is F32 [{heads}, rows, at most {dp}]; got {output.DType} {output.Shape}.", nameof(output));
        int rows = (int)output.Shape[1], outDim = (int)output.Shape[2];
        long rowsTotal = (long)heads * rows, total = rowsTotal * outDim;
        Int8Require(tileMax, DType.F32, rowsTotal * tiles, nameof(tileMax));
        Int8Require(tileSums, DType.U32, rowsTotal * tiles, nameof(tileSums));
        Int8Require(probabilities, DType.U8, rowsTotal * tiles * blockKeys, nameof(probabilities));
        Int8Require(valueScales, DType.F32, (long)kvHeads * (valueScalesPerTile ? Math.Max(1, tiles) : 1) * dp, nameof(valueScales));
        if (total == 0) return;
        if (keys == 0)
        {
            Fill(output, 0f);
            return;
        }
        RequireShader("int8_attn_u8_drain");

        using OpScope _op = EnterOp();
        VulkanBuffer p = GetBuffer(probabilities), v = GetBuffer(valueBytes), vs = GetBuffer(valueScales);
        VulkanBuffer tm = GetBuffer(tileMax), sm = GetBuffer(tileSums);
        VulkanBuffer outBuffer = _xfer.AllocateDevice((ulong)total * sizeof(float));
        try
        {
            VulkanKernel kernel = GetKernel("int8_attn_u8_drain", storageBufferCount: 6, _default1DSpec);
            Span<byte> pc = stackalloc byte[11 * 4];
            BinaryWriteUInt(pc, 0, (uint)rowsTotal);
            BinaryWriteUInt(pc, 4, (uint)rows);
            BinaryWriteUInt(pc, 8, (uint)(heads / kvHeads));
            BinaryWriteUInt(pc, 12, (uint)keys);
            BinaryWriteUInt(pc, 16, (uint)dp);
            BinaryWriteUInt(pc, 20, (uint)blockKeys);
            BinaryWriteUInt(pc, 24, (uint)tiles);
            BinaryWriteUInt(pc, 28, (uint)outDim);
            BinaryWriteUInt(pc, 32, valueScalesPerTile ? 1u : 0u);
            BinaryWriteFloat(pc, 36, Int8AttentionV2Reference.U8Offset);
            Span<ulong> buffers = stackalloc ulong[] { p.Handle, v.Handle, vs.Handle, tm.Handle, sm.Handle, outBuffer.Handle };
            for (long done = 0; done < total; done += MaxElementsPerDispatch)
            {
                BinaryWriteUInt(pc, 40, (uint)done);
                Dispatch(kernel, buffers, pc, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(output, outBuffer);
        }
        finally
        {
            _xfer.FreeDevice(outBuffer);
        }
    }

    /// <summary>INT8 attention with u8 probabilities over quantized operands: both passes. Unmasked, per-key K scales,
    /// one key segment, F32 output.</summary>
    /// <param name="output">Output, F32 <c>[heads, rows, outDim]</c>.</param>
    /// <param name="queryBytes">I8 <c>[heads · rows, Dp]</c>.</param>
    /// <param name="queryScales">F32, one a query row.</param>
    /// <param name="queryMeanDots">F32, one a query row, or null where the keys were not centered.</param>
    /// <param name="keyBytes">I8 <c>[kvHeads · keys, Dp]</c>.</param>
    /// <param name="keyScales">F32, one a key.</param>
    /// <param name="valueBytes">I8 <c>[kvHeads · keys, Dp]</c>.</param>
    /// <param name="valueScales">F32 <c>[kvHeads, tiles or 1, Dp]</c>.</param>
    /// <param name="heads">Query heads, the batch folded in.</param>
    /// <param name="kvHeads">KV heads, the batch folded in.</param>
    /// <param name="blockKeys">Keys a tile, 64 or 128.</param>
    /// <param name="valueScalesPerTile">One V scale a channel for each tile.</param>
    /// <param name="scale">The call's scale on the scores.</param>
    public void Int8AttentionU8(Tensor output, Tensor queryBytes, Tensor queryScales, Tensor? queryMeanDots, Tensor keyBytes,
        Tensor keyScales, Tensor valueBytes, Tensor valueScales, int heads, int kvHeads, int blockKeys, bool valueScalesPerTile,
        float scale)
    {
        (int rows, _, _, int tiles) = Int8U8Geometry(queryBytes, keyBytes, heads, kvHeads, blockKeys);
        long cells = (long)heads * rows * tiles;
        using Tensor tileMax = new(new TensorShape(cells), DType.F32);
        using Tensor tileSums = new(new TensorShape(cells), DType.U32);
        using Tensor probabilities = new(new TensorShape(cells * blockKeys), DType.U8);
        Int8AttentionU8Tiles(tileMax, tileSums, probabilities, queryBytes, queryScales, queryMeanDots, keyBytes, keyScales,
            heads, kvHeads, blockKeys, scale);
        Int8AttentionU8Drain(output, tileMax, tileSums, probabilities, valueBytes, valueScales, heads, kvHeads, blockKeys,
            valueScalesPerTile);
    }

    private static void Int8U8Heads(int heads, int kvHeads, int blockKeys)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(heads, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(kvHeads, 1);
        if (heads % kvHeads != 0)
            throw new ArgumentException($"{heads} query heads are not a whole multiple of {kvHeads} KV heads.", nameof(heads));
        if (blockKeys is not (64 or 128))
            throw new ArgumentOutOfRangeException(nameof(blockKeys), blockKeys, "A key tile is 64 or 128 keys.");
    }

    /// <summary>Query rows a head, keys a KV head, the padded head dim and the key tiles, from the byte operands.</summary>
    private static (int Rows, int Keys, int Dp, int Tiles) Int8U8Geometry(Tensor queryBytes, Tensor keyBytes, int heads,
        int kvHeads, int blockKeys)
    {
        Int8U8Heads(heads, kvHeads, blockKeys);
        if (queryBytes.DType != DType.I8 || queryBytes.Shape.Rank != 2 || keyBytes.DType != DType.I8 || keyBytes.Shape.Rank != 2
            || queryBytes.Shape[1] != keyBytes.Shape[1])
            throw new ArgumentException(
                $"The query and key bytes are I8 [rows, Dp] of one Dp; got {queryBytes.DType} {queryBytes.Shape} and {keyBytes.DType} {keyBytes.Shape}.");
        int dp = (int)queryBytes.Shape[1];
        if (dp is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(queryBytes), dp, "The padded head dim is 64, 128 or 256.");
        if (queryBytes.Shape[0] % heads != 0 || keyBytes.Shape[0] % kvHeads != 0)
            throw new ArgumentException(
                $"{queryBytes.Shape[0]} query rows over {heads} heads or {keyBytes.Shape[0]} keys over {kvHeads} KV heads is not a whole number.");
        int keys = (int)(keyBytes.Shape[0] / kvHeads);
        return ((int)(queryBytes.Shape[0] / heads), keys, dp, (keys + blockKeys - 1) / blockKeys);
    }

    private static void Int8Require(Tensor tensor, DType dtype, long elements, string name)
    {
        if (tensor.DType != dtype || tensor.ElementCount != elements)
            throw new ArgumentException($"{name} is {dtype} with {elements} elements; got {tensor.DType} {tensor.Shape}.", name);
    }
}
