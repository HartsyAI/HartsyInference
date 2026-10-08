namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Agreement measures between a tensor and its reference, accumulated in double.</summary>
internal static class DeepSeekV41VisionMetrics
{
    public static double RelL2(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)actual[i] - expected[i];
            diff += d * d;
            norm += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(diff / Math.Max(norm, 1e-30));
    }

    public static double Cosine(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            dot += (double)actual[i] * expected[i];
            na += (double)actual[i] * actual[i];
            nb += (double)expected[i] * expected[i];
        }
        return dot / Math.Max(Math.Sqrt(na * nb), 1e-30);
    }

    public static double MaxAbsDiff(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double max = 0;
        for (int i = 0; i < expected.Length; i++) max = Math.Max(max, Math.Abs((double)actual[i] - expected[i]));
        return max;
    }

    public static double MaxAbs(ReadOnlySpan<float> values)
    {
        double max = 0;
        foreach (float v in values) max = Math.Max(max, Math.Abs((double)v));
        return max;
    }
}
