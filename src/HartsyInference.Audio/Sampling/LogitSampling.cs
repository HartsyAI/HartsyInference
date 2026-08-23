namespace HartsyInference.Audio.Sampling;

/// <summary>Plain logit-to-token picks for AR decoders that do not need <see cref="NucleusSampler"/>'s
/// filtered-draw pipeline — greedy argmax and moshi's top-k temperature draw.</summary>
public static class LogitSampling
{
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
        // Indices of the k largest logits (k is small vs n; a full index sort is fine here).
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        float[] vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = logits[i];
        Array.Sort(idx, (a, b) => vals[b].CompareTo(vals[a]));   // descending by logit

        float max = vals[idx[0]] / temp;
        double sum = 0;
        double[] p = new double[k];
        for (int j = 0; j < k; j++) { p[j] = Math.Exp(vals[idx[j]] / temp - max); sum += p[j]; }
        double r = rng.NextDouble() * sum, acc = 0;
        for (int j = 0; j < k; j++) { acc += p[j]; if (r <= acc) return idx[j]; }
        return idx[k - 1];
    }
}
