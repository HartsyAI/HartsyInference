using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Duo;

/// <summary>Row order of the fused <c>qkv_proj</c>. The released backbone shards store it per head, <c>[head0: q k v, head1: q k v, ...]</c> (each part <c>headDim</c> rows); the engine's <c>QkvSplitNormHeadMajor</c> reads three <c>[heads*headDim]</c> blocks, <c>[Q; K; V]</c>.
/// Upstream does the same reorder at load time (<c>reorder_interleaved_qkv</c>). Rows move whole, so any dtype works and nothing is rounded.</summary>
public static class DuoQkvLayout
{
    /// <summary>Per-head interleaved to <c>[Q; K; V]</c> blocks. Returns a new tensor the caller owns.</summary>
    public static Tensor ToBlockOrder(Tensor interleaved, int heads, int headDim) => Permute(interleaved, heads, headDim, toBlocks: true);

    /// <summary><c>[Q; K; V]</c> blocks to per-head interleaved. Returns a new tensor the caller owns.</summary>
    public static Tensor ToHeadInterleaved(Tensor blocks, int heads, int headDim) => Permute(blocks, heads, headDim, toBlocks: false);

    private static unsafe Tensor Permute(Tensor source, int heads, int headDim, bool toBlocks)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (heads <= 0 || headDim <= 0) throw new ArgumentOutOfRangeException(nameof(heads), "Heads and head dim must be positive.");
        if (source.DType.IsQuantized) throw new NotSupportedException($"A {source.DType} qkv weight cannot be reordered row by row.");
        long rows = 3L * heads * headDim;
        if (source.Shape.Rank != 2 || source.Shape[0] != rows)
            throw new ArgumentException($"A fused qkv weight must be [{rows}, in], got {source.Shape}.", nameof(source));

        long rowBytes = source.Shape[1] * source.DType.SizeInBytes;
        Tensor result = new(source.Shape, source.DType);
        byte* src = (byte*)source.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        for (int part = 0; part < 3; part++)
        {
            for (int head = 0; head < heads; head++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    long interleavedRow = ((long)head * 3 + part) * headDim + d;
                    long blockRow = ((long)part * heads + head) * headDim + d;
                    long from = toBlocks ? interleavedRow : blockRow, to = toBlocks ? blockRow : interleavedRow;
                    Buffer.MemoryCopy(src + from * rowBytes, dst + to * rowBytes, rowBytes, rowBytes);
                }
            }
        }
        return result;
    }
}
