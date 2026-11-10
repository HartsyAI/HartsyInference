using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Sparse-attention index building: the per-query top-k over indexer scores and the merge of that top-k with a trailing window.
public partial interface IBackend
{
    /// <summary>Per row of <paramref name="scores"/>, the <c>k = min(topK, cols)</c> best columns in ascending column order; a column
    /// at or past <c>limits[row]</c> becomes -1, any other gains <paramref name="offset"/>. See <see cref="Host.SparseTopKReference"/>.</summary>
    /// <remarks>Best means largest under <see cref="float.CompareTo(float)"/>: NaN ranks below every number, -0 equals +0, and ties
    /// keep the lower column. The selection is exact, so every backend returns the same indices for the same scores.</remarks>
    /// <param name="indices">I32 <c>[rows, k]</c>.</param>
    /// <param name="scores">F32 <c>[rows, cols]</c>.</param>
    /// <param name="limits">I32 <c>[rows]</c>.</param>
    void SparseTopK(Tensor indices, Tensor scores, Tensor limits, int topK, int offset) =>
        throw NotSupportedPrimitive(nameof(SparseTopK));

    /// <summary>Per query, the row <c>[topk[:n] | seq-w .. seq-1 | -1 ...]</c> with <c>seq = positions[t] + 1</c>,
    /// <c>n = min(max(seq - window, 0), topkCols)</c> and <c>w = min(seq, window)</c>: an indexer top-k that skips the trailing
    /// window, then that window. See <see cref="Host.WindowTopKMergeReference"/>.</summary>
    /// <param name="rows">I32 <c>[tokens, width]</c> with <c>width >= topkCols + window</c>.</param>
    /// <param name="topk">I32 <c>[tokens, topkCols]</c>; entries are copied as given.</param>
    /// <param name="positions">I32 <c>[tokens]</c> absolute query positions, non-negative.</param>
    void MergeWindowTopK(Tensor rows, Tensor topk, Tensor positions, int window) =>
        throw NotSupportedPrimitive(nameof(MergeWindowTopK));
}
