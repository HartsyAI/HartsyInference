using HartsyInference.Audio.Sampling;
using Xunit;

namespace HartsyInference.Audio.Tests.Sampling;

/// <summary>Proves <see cref="LogitSampling.SampleTopK"/> (Kyutai's text-token and depformer-codebook
/// sampling — up to 32 calls per audio frame) still returns the EXACT token the pre-fix algorithm did, for the
/// same logits/seed, after swapping its per-call delegate-comparator full sort for the shared primitive-sort
/// scratch buffers. <see cref="ReferenceSampleTopK"/> is the pre-fix algorithm kept verbatim as the oracle.</summary>
public sealed class LogitSamplingIdentityTests
{
    /// <summary>Verbatim pre-fix <c>LogitSampling.SampleTopK</c>, kept ONLY as the test oracle.</summary>
    private static int ReferenceSampleTopK(ReadOnlySpan<float> logits, float temp, int topK, Random rng)
    {
        int n = logits.Length;
        int k = topK <= 0 ? n : Math.Min(topK, n);
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        float[] vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = logits[i];
        Array.Sort(idx, (a, b) => vals[b].CompareTo(vals[a]));

        float max = vals[idx[0]] / temp;
        double sum = 0;
        double[] p = new double[k];
        for (int j = 0; j < k; j++) { p[j] = Math.Exp(vals[idx[j]] / temp - max); sum += p[j]; }
        double r = rng.NextDouble() * sum, acc = 0;
        for (int j = 0; j < k; j++) { acc += p[j]; if (r <= acc) return idx[j]; }
        return idx[k - 1];
    }

    private static float[] MakeLogits(int n, Random rng)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
        return v;
    }

    /// <summary>Two consecutive calls from two <see cref="Random"/> instances seeded identically must match on
    /// both the returned token and whether the SECOND call also matches — proving <c>SampleTopK</c> consumes
    /// exactly one <c>NextDouble()</c> per call, same as before.</summary>
    private static void AssertIdentical(float[] logits, float temp, int topK, int seed, string label)
    {
        Random rngOld = new(seed);
        Random rngNew = new(seed);

        int tokenOld1 = ReferenceSampleTopK(logits, temp, topK, rngOld);
        int tokenNew1 = LogitSampling.SampleTopK(logits, temp, topK, rngNew);
        Assert.True(tokenOld1 == tokenNew1, $"{label}: first call old={tokenOld1} new={tokenNew1}");

        int tokenOld2 = ReferenceSampleTopK(logits, temp, topK, rngOld);
        int tokenNew2 = LogitSampling.SampleTopK(logits, temp, topK, rngNew);
        Assert.True(tokenOld2 == tokenNew2, $"{label}: second call (rng-advance check) old={tokenOld2} new={tokenNew2}");
    }

    [Theory]
    [InlineData(2048, 250)] // Mimi/depformer codebook cardinality, doc-cited top-k
    [InlineData(8000, 50)]  // Kyutai text cardinality (TextCard)
    [InlineData(1, 250)]
    [InlineData(2048, 1)]   // k=1 -> effectively argmax-of-the-draw
    [InlineData(2048, 2048)] // k == n
    [InlineData(2048, 0)]   // topK<=0 -> k=n (unbounded)
    [InlineData(2048, 5000)] // topK > n, clamps to n
    public void RandomLogits_MatchesReference(int n, int topK)
    {
        Random rng = new(10_000 + n + topK);
        for (int trial = 0; trial < 15; trial++)
        {
            float[] logits = MakeLogits(n, rng);
            AssertIdentical(logits, temp: 0.8f, topK, seed: 1 + trial, label: $"n={n} k={topK} trial={trial}");
        }
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(0.8f)]
    [InlineData(1.0f)]
    [InlineData(5.0f)]
    public void TemperatureSweep_MatchesReference(float temp)
    {
        Random rng = new(2024);
        float[] logits = MakeLogits(2048, rng);
        AssertIdentical(logits, temp, topK: 250, seed: 3, label: $"temp={temp}");
    }

    /// <summary>Exact ties at the top: several ids share the identical maximal logit, which both decides
    /// top-k MEMBERSHIP at a boundary and, by sharing the maximal value, forces several identical terms through
    /// the softmax/draw.</summary>
    [Theory]
    [InlineData(2)]
    [InlineData(10)]
    [InlineData(300)] // more ties than k=250
    public void ExactTiesAtTop_MatchesReference(int numTied)
    {
        int n = 2048, topK = 250;
        Random rng = new(4000 + numTied);
        float[] logits = MakeLogits(n, rng);
        for (int i = 0; i < Math.Min(numTied, n); i++) logits[i] = 7.0f;
        for (int trial = 0; trial < 10; trial++)
        {
            AssertIdentical(logits, 0.8f, topK, seed: 100 + trial, label: $"numTied={numTied} trial={trial}");
        }
    }
}
