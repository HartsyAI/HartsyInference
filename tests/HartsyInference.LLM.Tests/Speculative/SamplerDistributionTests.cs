using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculative;

public sealed class SamplerDistributionTests
{
    private static readonly float[] Logits = [1.2f, -0.4f, 2.0f, 0.3f, 1.1f, -1.5f, 0.9f, 2.4f];

    [Fact]
    public void Distribution_Matches_The_Draws_Next_Makes()
    {
        SamplingOptions options = new() { Temperature = 0.7f, TopK = 5, TopP = 0.95f, Seed = 17 };
        SamplerChain expectedChain = SamplerChain.FromOptions(options);
        float[] expected = new float[Logits.Length];
        expectedChain.Distribution(Logits.AsSpan().ToArray(), [], expected);

        SamplerChain chain = SamplerChain.FromOptions(options);
        long[] counts = new long[Logits.Length];
        float[] buffer = new float[Logits.Length];
        for (int i = 0; i < 300_000; i++)
        {
            Logits.CopyTo(buffer, 0);
            counts[chain.Next(buffer, [])]++;
        }
        double stat = SpeculativeTestSupport.ChiSquare(counts, expected.Select(p => (double)p).ToArray(), out int df);
        Assert.True(stat < SpeculativeTestSupport.Critical(df), $"chi-square {stat:F2} over {df} df");
    }

    [Fact]
    public void Excluded_Tokens_Have_Zero_Probability()
    {
        SamplerChain chain = SamplerChain.FromOptions(new SamplingOptions { TopK = 3 });
        float[] probs = new float[Logits.Length];
        chain.Distribution(Logits.AsSpan().ToArray(), [], probs);
        Assert.Equal(3, probs.Count(p => p > 0));
        Assert.Equal(1f, probs.Sum(), 5);
    }
}
