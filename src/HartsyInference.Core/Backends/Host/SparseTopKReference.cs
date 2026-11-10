using System.Buffers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>CPU reference for <see cref="IBackend.SparseTopK"/>: an exact per-row top-k returned in ascending column order.</summary>
public static class SparseTopKReference
{
    /// <summary>Checks the operands and returns the rows, the columns and <c>k = min(topK, cols)</c>.</summary>
    public static (int Rows, int Cols, int K) Validate(Tensor indices, Tensor scores, Tensor limits, int topK)
    {
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(limits);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);
        if (scores.DType != DType.F32 || scores.Shape.Rank != 2)
            throw new ArgumentException($"SparseTopK scores must be F32 [rows, cols]; got {scores.Shape} {scores.DType.Name}.", nameof(scores));
        long rows = scores.Shape[0], cols = scores.Shape[1];
        if (rows > int.MaxValue || cols > int.MaxValue || rows * cols > int.MaxValue)
            throw new NotSupportedException($"SparseTopK addresses at most {int.MaxValue} scores; got {scores.Shape}.");
        int k = (int)Math.Min(topK, cols);
        if (limits.DType != DType.I32 || limits.ElementCount != rows)
            throw new ArgumentException($"SparseTopK limits must be I32 with {rows} entries; got {limits.Shape} {limits.DType.Name}.",
                nameof(limits));
        if (indices.DType != DType.I32 || indices.Shape.Rank != 2 || indices.Shape[0] != rows || indices.Shape[1] != k)
            throw new ArgumentException($"SparseTopK indices must be I32 [{rows}, {k}]; got {indices.Shape} {indices.DType.Name}.", nameof(indices));
        return ((int)rows, (int)cols, k);
    }

    /// <summary>Writes every row's selection into <paramref name="indices"/>.</summary>
    public static void Apply(Tensor indices, Tensor scores, Tensor limits, int topK, int offset)
    {
        (int rows, int cols, int k) = Validate(indices, scores, limits, topK);
        ReadOnlySpan<float> all = scores.AsReadOnlySpan<float>();
        ReadOnlySpan<int> lim = limits.AsReadOnlySpan<int>();
        Span<int> dest = indices.AsSpan<int>();
        for (int r = 0; r < rows; r++) SelectRow(all.Slice(r * cols, cols), lim[r], k, offset, dest.Slice(r * k, k));
    }

    /// <summary>The <paramref name="k"/> best positions of one row in ascending order, shifted by <paramref name="offset"/>; positions
    /// at or past <paramref name="limit"/> become -1, so the first <c>min(limit, k)</c> entries are the valid ones when every
    /// reachable score is a number.</summary>
    public static void SelectRow(ReadOnlySpan<float> scores, int limit, int k, int offset, Span<int> dest)
    {
        if ((uint)k > (uint)scores.Length) throw new ArgumentOutOfRangeException(nameof(k));
        if (dest.Length < k) throw new ArgumentException("dest is too small.", nameof(dest));
        int[] order = ArrayPool<int>.Shared.Rent(scores.Length);
        float[] values = ArrayPool<float>.Shared.Rent(scores.Length);
        try
        {
            Span<int> idx = order.AsSpan(0, scores.Length);
            for (int i = 0; i < idx.Length; i++) idx[i] = i;
            scores.CopyTo(values);
            idx.Sort(new BestFirst(values));
            Span<int> picked = idx[..k];
            picked.Sort();
            for (int i = 0; i < k; i++) dest[i] = picked[i] < limit ? picked[i] + offset : -1;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(order);
            ArrayPool<float>.Shared.Return(values);
        }
    }

    // Larger value first under float.CompareTo (NaN lowest, -0 == +0), then the lower index: a total order, so any sort agrees.
    private readonly struct BestFirst(float[] values) : IComparer<int>
    {
        public int Compare(int a, int b)
        {
            int c = values[b].CompareTo(values[a]);
            return c != 0 ? c : a.CompareTo(b);
        }
    }
}
