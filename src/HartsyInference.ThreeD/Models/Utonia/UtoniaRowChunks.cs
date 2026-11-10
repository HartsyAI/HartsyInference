using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Runs a row-wise operator over a point set in row chunks. The full-width result lives in host memory, so a level's
/// features never occupy VRAM whole; each chunk's device temporaries are bounded by the chunk budget.
/// </summary>
internal static class UtoniaRowChunks
{
    /// <summary>Default byte budget of one chunk's widest operand.</summary>
    internal const long ChunkBytes = 64L << 20;

    /// <summary>Fills <paramref name="result"/> from the rows of <paramref name="source"/> held by one chunk.</summary>
    internal delegate void RowOperator(Tensor source, Tensor result);

    /// <summary>Two-input form of <see cref="RowOperator"/>; both inputs carry the same rows.</summary>
    internal delegate void RowOperator2(Tensor first, Tensor second, Tensor result);

    /// <summary>Rows per chunk so that one row of <paramref name="width"/> floats fits the budget; at least one row.</summary>
    public static int RowsPerChunk(long chunkBytes, long width) => (int)Math.Max(1, chunkBytes / (sizeof(float) * width));

    /// <summary>Applies <paramref name="apply"/> chunk by chunk to <paramref name="source"/>, returning a host <c>[rows, outChannels]</c> tensor.</summary>
    public static Tensor Map(Tensor source, long outChannels, int rowsPerChunk, RowOperator apply)
    {
        long n = source.Shape[0];
        Tensor output = new(new TensorShape(n, outChannels), DType.F32);
        Span<float> dst = output.AsSpan<float>();
        for (long start = 0; start < n; start += rowsPerChunk)
        {
            int rows = (int)Math.Min(rowsPerChunk, n - start);
            using Tensor src = source.SliceRows(start, rows);
            using Tensor res = new(new TensorShape(rows, outChannels), DType.F32);
            apply(src, res);
            res.AsReadOnlySpan<float>().CopyTo(dst.Slice((int)(start * outChannels), rows * (int)outChannels));
        }

        return output;
    }

    /// <summary>Two-input <see cref="Map"/>: the inputs must have equal row counts.</summary>
    public static Tensor Map2(Tensor first, Tensor second, long outChannels, int rowsPerChunk, RowOperator2 apply)
    {
        long n = first.Shape[0];
        Tensor output = new(new TensorShape(n, outChannels), DType.F32);
        Span<float> dst = output.AsSpan<float>();
        for (long start = 0; start < n; start += rowsPerChunk)
        {
            int rows = (int)Math.Min(rowsPerChunk, n - start);
            using Tensor a = first.SliceRows(start, rows);
            using Tensor b = second.SliceRows(start, rows);
            using Tensor res = new(new TensorShape(rows, outChannels), DType.F32);
            apply(a, b, res);
            res.AsReadOnlySpan<float>().CopyTo(dst.Slice((int)(start * outChannels), rows * (int)outChannels));
        }

        return output;
    }
}
