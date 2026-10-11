using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary><see cref="TopKStep"/> finds the k-th largest logit with a bounded selection instead of sorting the vocabulary. The kept set must be
/// exactly what the sort-based threshold gave, ties and masked entries included.</summary>
public sealed class TopKSelectionIdentityTests
{
    private static void ReferenceApply(float[] logits, int k)
    {
        if (k <= 0 || k >= logits.Length) return;
        float[] sorted = (float[])logits.Clone();
        Array.Sort(sorted);
        float threshold = sorted[logits.Length - k];
        int kept = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] >= threshold && kept < k) kept++;
            else logits[i] = float.NegativeInfinity;
        }
    }

    [Theory]
    [InlineData(1, 1000, 1)]
    [InlineData(2, 50304, 40)]
    [InlineData(3, 151936, 40)]
    [InlineData(4, 2000, 1999)]
    [InlineData(5, 4096, 100)]
    public void KeptSet_MatchesTheSortBasedThreshold(int seed, int vocab, int k)
    {
        Random rng = new(seed);
        for (int round = 0; round < 6; round++)
        {
            float[] logits = new float[vocab];
            for (int i = 0; i < vocab; i++) logits[i] = (float)(rng.NextDouble() * 20 - 10);
            if (round % 2 == 1)
            {
                // Heavy ties and masked entries: quantize the values and mask a slice, as repetition penalty and grammars do.
                for (int i = 0; i < vocab; i++) logits[i] = MathF.Round(logits[i]);
                for (int i = 0; i < vocab / 5; i++) logits[rng.Next(vocab)] = float.NegativeInfinity;
            }
            float[] expected = (float[])logits.Clone();
            ReferenceApply(expected, k);
            float[] actual = (float[])logits.Clone();
            new TopKStep(k).Apply(actual, []);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Step_IsReusableAcrossTokens()
    {
        Random rng = new(11);
        TopKStep step = new(40);
        for (int token = 0; token < 20; token++)
        {
            float[] logits = new float[8192];
            for (int i = 0; i < logits.Length; i++) logits[i] = (float)(rng.NextDouble() * 12 - 6);
            float[] expected = (float[])logits.Clone();
            ReferenceApply(expected, 40);
            step.Apply(logits, []);
            Assert.Equal(expected, logits);
        }
    }
}
