namespace HartsyInference.Audio.Dsp;

/// <summary>Joins consecutive waves with a linear cross-fade, as the F5-TTS and ZipVoice long-text paths do.</summary>
public static class WaveCrossFade
{
    /// <summary>Joins the waves in order; each join overlaps <paramref name="crossFadeSamples"/> with a linear fade,
    /// clamped to the shorter wave, and a join with no overlap concatenates.</summary>
    public static float[] Join(IReadOnlyList<float[]> waves, int crossFadeSamples)
    {
        if (waves.Count == 0) throw new ArgumentException("At least one wave is required.", nameof(waves));
        float[] joined = waves[0];
        for (int i = 1; i < waves.Count; i++) joined = JoinPair(joined, waves[i], crossFadeSamples);
        return joined;
    }

    private static float[] JoinPair(float[] prev, float[] next, int crossFadeSamples)
    {
        int n = Math.Min(crossFadeSamples, Math.Min(prev.Length, next.Length));
        if (n <= 0)
        {
            float[] concatenated = new float[prev.Length + next.Length];
            prev.CopyTo(concatenated, 0);
            next.CopyTo(concatenated, prev.Length);
            return concatenated;
        }
        int head = prev.Length - n;
        float[] outp = new float[prev.Length + next.Length - n];
        Array.Copy(prev, 0, outp, 0, head);
        for (int k = 0; k < n; k++)
        {
            double fadeIn = n == 1 ? 0.0 : k / (double)(n - 1);
            outp[head + k] = (float)(prev[head + k] * (1.0 - fadeIn) + next[k] * fadeIn);
        }
        Array.Copy(next, n, outp, head + n, next.Length - n);
        return outp;
    }
}
