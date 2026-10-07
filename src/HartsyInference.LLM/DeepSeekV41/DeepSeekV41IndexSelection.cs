namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Per-query selection over indexer scores: level-one candidate blocks and the final sorted top-k of compressed positions.</summary>
/// <remarks>Host reference for upstream <c>select_candidate_blocks</c> and the tail of <c>Indexer.forward</c>. Ties break toward
/// the lower index, so results are deterministic where torch leaves tie order unspecified.</remarks>
public static class DeepSeekV41IndexSelection
{
    /// <summary>Marks the positions of the <paramref name="topkBlocks"/> best blocks (each scored by its best position) for one query.</summary>
    /// <param name="logits">Scores per compressed position, unreachable ones at <c>-inf</c>.</param>
    /// <param name="compressLen">How many compressed positions this query can reach; the block holding the newest is always kept.</param>
    /// <param name="topkBlocks">Blocks to keep.</param>
    /// <param name="blockSize">Positions per block.</param>
    /// <param name="mask">Receives one flag per position; same length as <paramref name="logits"/>.</param>
    public static void SelectCandidateBlocks(ReadOnlySpan<float> logits, int compressLen, int topkBlocks, int blockSize, Span<bool> mask)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(blockSize, 1);
        if (mask.Length != logits.Length) throw new ArgumentException("mask must match logits in length.", nameof(mask));
        int width = logits.Length;
        int blocks = (width + blockSize - 1) / blockSize;
        float[] scores = new float[blocks];
        for (int b = 0; b < blocks; b++)
        {
            float best = float.NegativeInfinity;
            int end = Math.Min(width, (b + 1) * blockSize);
            for (int i = b * blockSize; i < end; i++) best = MathF.Max(best, logits[i]);
            scores[b] = best;
        }
        if (compressLen > 0) scores[Math.Min((compressLen - 1) / blockSize, blocks - 1)] = float.PositiveInfinity;

        bool[] keep = new bool[blocks];
        int[] order = TopIndices(scores, Math.Min(topkBlocks, blocks));
        foreach (int b in order) keep[b] = scores[b] > float.NegativeInfinity;
        for (int i = 0; i < width; i++) mask[i] = keep[i / blockSize];
    }

    /// <summary>The <paramref name="indexTopk"/> best positions in ascending order, shifted by <paramref name="offset"/>; unreachable ones become -1.</summary>
    /// <param name="scores">Indexer scores per compressed position for one query.</param>
    /// <param name="compressLen">Positions at or beyond this are unreachable.</param>
    /// <param name="indexTopk">Configured top-k; clamped to the number of positions.</param>
    /// <param name="offset">Added to kept positions so they index after the window slots.</param>
    /// <param name="dest">Receives <c>min(indexTopk, scores.Length)</c> entries.</param>
    /// <returns>The number of entries written.</returns>
    public static int SelectTopK(ReadOnlySpan<float> scores, int compressLen, int indexTopk, int offset, Span<int> dest)
    {
        int k = Math.Min(indexTopk, scores.Length);
        if (dest.Length < k) throw new ArgumentException("dest is too small.", nameof(dest));
        int[] picked = TopIndices(scores.ToArray(), k);
        Array.Sort(picked);
        for (int i = 0; i < k; i++) dest[i] = picked[i] < compressLen ? picked[i] + offset : -1;
        return k;
    }

    private static int[] TopIndices(float[] values, int k)
    {
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
