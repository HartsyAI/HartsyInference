using HartsyInference.Core.Backends.Host;

namespace HartsyInference.LLM.Transformer.Sparse;

/// <summary>Builds one query's sparse key set from an indexer top-k that skips the trailing window, followed by that window
/// (vLLM-ERNIE <c>merge_swa_topk</c>).</summary>
/// <remarks>With <c>seq = position + 1</c>, <c>n = min(max(seq - window, 0), topk)</c> and <c>w = min(seq, window)</c> the row is
/// <c>[topk[:n] | seq-w .. seq-1 | -1 ...]</c>. The indexer only scores keys before <c>seq - window</c>, so the two parts never overlap.</remarks>
public static class SparseWindowMerge
{
    /// <summary>Indexer keys a query at <paramref name="position"/> may score: those before its trailing window.</summary>
    public static int IndexerLimit(int position, int window) => Math.Max(position + 1 - window, 0);

    /// <summary>Writes the merged row for one query and returns how many leading entries are valid (the rest are -1).</summary>
    /// <param name="topk">The indexer's ascending top-k for this query; its first <c>n</c> entries must be valid.</param>
    /// <param name="topkWidth">The configured top-k, <c>n</c>'s cap.</param>
    /// <param name="position">The query's absolute position.</param>
    /// <param name="window">Sliding-window width.</param>
    /// <param name="row">Receives the row; at least <c>topkWidth + window</c> wide.</param>
    public static int Merge(ReadOnlySpan<int> topk, int topkWidth, int position, int window, Span<int> row)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);
        int n = Math.Min(Math.Max(position + 1 - window, 0), Math.Max(topkWidth, 0));
        for (int i = 0; i < n && i < topk.Length; i++)
        {
            if ((uint)topk[i] >= (uint)(position + 1 - window))
                throw new ArgumentException($"topk entry {topk[i]} is outside [0, {position + 1 - window}).", nameof(topk));
        }
        return WindowTopKMergeReference.MergeRow(topk, topkWidth, position, window, row);
    }
}
