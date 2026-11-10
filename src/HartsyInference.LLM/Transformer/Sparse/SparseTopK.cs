using HartsyInference.Core.Backends.Host;

namespace HartsyInference.LLM.Transformer.Sparse;

/// <summary>Per-query top-k over indexer scores, shared by the sparse-attention layers (DeepSeek V4.1 compressed positions, DSA tokens);
/// the device form is <see cref="HartsyInference.Core.Backends.IBackend.SparseTopK"/>.</summary>
/// <remarks>Ties break toward the lower index, so results are deterministic where torch leaves tie order unspecified.</remarks>
public static class SparseTopK
{
    /// <summary>The <paramref name="topk"/> best positions in ascending order, shifted by <paramref name="offset"/>; positions at or
    /// past <paramref name="limit"/> become -1, so the first <c>min(limit, k)</c> entries are the valid ones.</summary>
    /// <param name="scores">Scores per position for one query; unreachable ones at <c>-inf</c>.</param>
    /// <param name="limit">Positions at or beyond this are unreachable.</param>
    /// <param name="topk">Configured top-k; clamped to the number of positions.</param>
    /// <param name="offset">Added to kept positions.</param>
    /// <param name="dest">Receives <c>min(topk, scores.Length)</c> entries.</param>
    /// <returns>The number of entries written.</returns>
    public static int SelectSorted(ReadOnlySpan<float> scores, int limit, int topk, int offset, Span<int> dest)
    {
        int k = Math.Min(topk, scores.Length);
        if (dest.Length < k) throw new ArgumentException("dest is too small.", nameof(dest));
        SparseTopKReference.SelectRow(scores, limit, k, offset, dest);
        return k;
    }

    /// <summary>The indices of the <paramref name="k"/> largest values, best first, ties toward the lower index.</summary>
    public static int[] Best(float[] values, int k)
    {
        ArgumentNullException.ThrowIfNull(values);
        if ((uint)k > (uint)values.Length) throw new ArgumentOutOfRangeException(nameof(k));
        int[] idx = new int[values.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) =>
        {
            int c = values[b].CompareTo(values[a]);
            return c != 0 ? c : a.CompareTo(b);
        });
        return idx.AsSpan(0, k).ToArray();
    }
}
