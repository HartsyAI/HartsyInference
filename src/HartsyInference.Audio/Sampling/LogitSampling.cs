namespace HartsyInference.Audio.Sampling;

/// <summary>Plain logit-to-token picks for AR decoders that do not need <see cref="NucleusSampler"/>'s
/// filtered-draw pipeline — greedy argmax and moshi's top-k temperature draw.</summary>
public static class LogitSampling
{
    // SampleTopK scratch, reused across calls on the same thread (one generation's decode loop samples
    // synchronously on a single thread) — grown, never shrunk. See NucleusSampler's fallback-path remarks for
    // why this is thread-static rather than an instance field: SampleTopK is itself a stateless static call.
    [ThreadStatic] private static int[]? t_idx;
    [ThreadStatic] private static float[]? t_vals;
    [ThreadStatic] private static double[]? t_p;

    /// <summary>Index of the largest logit; ties resolve to the lowest index.</summary>
    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    /// <summary>Top-k temperature sampling (moshi <c>sample_token</c>): scale logits by <paramref name="temp"/>,
    /// keep the <paramref name="topK"/> highest, softmax over them, then multinomial-sample with
    /// <paramref name="rng"/>. Kyutai TTS was trained for sampling (audio temp 0.8 / top-k 250); greedy argmax
    /// collapses the code cascade to non-speech.</summary>
    public static int SampleTopK(ReadOnlySpan<float> logits, float temp, int topK, Random rng)
    {
        int n = logits.Length;
        int k = topK <= 0 ? n : Math.Min(topK, n);
        // Was a fresh int[n] + float[n] plus a full delegate-comparator Array.Sort every call, regardless of
        // how small k was relative to n — the same anti-pattern PR #215 fixed in the LLM package. n here is at
        // most Kyutai's text/codec cardinality (thousands), so a full sort is still cheap in absolute terms,
        // but this runs once per codebook (up to 32x) per audio frame, so the allocations were not.
        if (t_idx is null || t_idx.Length < n) t_idx = new int[n];
        if (t_vals is null || t_vals.Length < n) t_vals = new float[n];
        int[] idx = t_idx;
        float[] vals = t_vals;
        for (int i = 0; i < n; i++) { idx[i] = i; vals[i] = logits[i]; }
        SortHelpers.SortDescendingByValue(vals, idx, n);   // descending by logit
        // vals[j] is now the j-th largest logit; idx[j] is that logit's original token id.

        float max = vals[0] / temp;
        if (t_p is null || t_p.Length < k) t_p = new double[k];
        double[] p = t_p;
        double sum = 0;
        for (int j = 0; j < k; j++) { p[j] = Math.Exp(vals[j] / temp - max); sum += p[j]; }
        double r = rng.NextDouble() * sum, acc = 0;
        for (int j = 0; j < k; j++) { acc += p[j]; if (r <= acc) return idx[j]; }
        return idx[k - 1];
    }
}
