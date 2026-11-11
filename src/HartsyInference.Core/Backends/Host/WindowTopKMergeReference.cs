using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>CPU reference for <see cref="IBackend.MergeWindowTopK"/>: an indexer top-k that skips the trailing window, followed by
/// that window (vLLM-ERNIE <c>merge_swa_topk</c>).</summary>
public static class WindowTopKMergeReference
{
    /// <summary>Checks the operands and returns the tokens, the top-k columns and the row width.</summary>
    public static (int Tokens, int TopkCols, int Width) Validate(Tensor rows, Tensor topk, Tensor positions, int window)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(topk);
        ArgumentNullException.ThrowIfNull(positions);
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);
        if (topk.DType != DType.I32 || topk.Shape.Rank != 2)
            throw new ArgumentException($"MergeWindowTopK topk must be I32 [tokens, cols]; got {topk.Shape} {topk.DType.Name}.", nameof(topk));
        long tokens = topk.Shape[0], cols = topk.Shape[1];
        if (positions.DType != DType.I32 || positions.ElementCount != tokens)
            throw new ArgumentException($"MergeWindowTopK positions must be I32 with {tokens} entries.", nameof(positions));
        if (rows.DType != DType.I32 || rows.Shape.Rank != 2 || rows.Shape[0] != tokens || rows.Shape[1] < cols + window)
            throw new ArgumentException($"MergeWindowTopK rows must be I32 [{tokens}, >= {cols + window}]; got {rows.Shape}.", nameof(rows));
        if (rows.ElementCount > int.MaxValue || topk.ElementCount > int.MaxValue)
            throw new NotSupportedException("MergeWindowTopK operands exceed 32-bit indexing.");
        return ((int)tokens, (int)cols, (int)rows.Shape[1]);
    }

    /// <summary>Writes every query's merged row.</summary>
    public static void Apply(Tensor rows, Tensor topk, Tensor positions, int window)
    {
        (int tokens, int cols, int width) = Validate(rows, topk, positions, window);
        ReadOnlySpan<int> top = topk.AsReadOnlySpan<int>(), pos = positions.AsReadOnlySpan<int>();
        Span<int> dest = rows.AsSpan<int>();
        for (int t = 0; t < tokens; t++) MergeRow(top.Slice(t * cols, cols), cols, pos[t], window, dest.Slice(t * width, width));
    }

    /// <summary>Writes one merged row and returns how many leading entries it holds (the rest are -1). The first
    /// <c>min(max(seq - window, 0), topkWidth)</c> entries of <paramref name="topk"/> are copied as given.</summary>
    public static int MergeRow(ReadOnlySpan<int> topk, int topkWidth, int position, int window, Span<int> row)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(topkWidth);
        if (row.Length < topkWidth + window) throw new ArgumentException("row must hold topkWidth + window entries.", nameof(row));
        int seq = position + 1;
        int n = Math.Min(Math.Max(seq - window, 0), topkWidth);
        int w = Math.Min(seq, window);
        if (topk.Length < n) throw new ArgumentException($"topk holds {topk.Length} entries; this query needs {n}.", nameof(topk));
        topk[..n].CopyTo(row);
        for (int i = 0; i < w; i++) row[n + i] = seq - w + i;
        row[(n + w)..].Fill(-1);
        return n + w;
    }
}
