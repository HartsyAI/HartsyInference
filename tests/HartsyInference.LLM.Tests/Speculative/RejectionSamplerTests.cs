using HartsyInference.LLM.Generation.Speculative;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculative;

public sealed class RejectionSamplerTests
{
    private static float[] OneHot(int vocab, int at)
    {
        float[] row = new float[vocab];
        row[at] = 1f;
        return row;
    }

    [Fact]
    public void Greedy_Target_Accepts_Exactly_The_Argmax_Draft()
    {
        const int vocab = 6;
        float[][] target = [OneHot(vocab, 4), OneHot(vocab, 1), OneHot(vocab, 5), OneHot(vocab, 0)];
        Func<double> uniform = SpeculativeTestSupport.Uniform(3);

        SpeculativeOutcome all = RejectionSampler.Verify([4, 1, 5], target, null, uniform);
        Assert.Equal(new SpeculativeOutcome(3, 0), all);

        SpeculativeOutcome second = RejectionSampler.Verify([4, 2, 5], target, null, uniform);
        Assert.Equal(new SpeculativeOutcome(1, 1), second);

        SpeculativeOutcome first = RejectionSampler.Verify([3, 1, 5], target, null, uniform);
        Assert.Equal(new SpeculativeOutcome(0, 4), first);
    }

    [Fact]
    public void Stochastic_Proposal_Emits_The_Target_Distribution_Exactly()
    {
        float[] p = [0.05f, 0.30f, 0.0f, 0.15f, 0.10f, 0.0f, 0.25f, 0.15f];
        float[] q = [0.20f, 0.05f, 0.25f, 0.10f, 0.05f, 0.15f, 0.10f, 0.10f];
        Func<double> draw = SpeculativeTestSupport.Uniform(41);
        Func<double> accept = SpeculativeTestSupport.Uniform(42);
        long[] counts = new long[p.Length];
        const int trials = 1_000_000;

        for (int t = 0; t < trials; t++)
        {
            int x = Inverse(q, draw());
            SpeculativeOutcome outcome = RejectionSampler.Verify([x], [p, p], [q], accept);
            counts[outcome.Accepted == 1 ? x : outcome.NextToken]++;
        }
        AssertMatches(counts, p);
    }

    [Fact]
    public void Deterministic_Proposal_Emits_The_Target_Distribution_Exactly()
    {
        // whatever the proposer guesses, the emitted token must still follow the target
        float[] p = [0.10f, 0.02f, 0.28f, 0.20f, 0.0f, 0.15f, 0.05f, 0.20f];
        Func<double> guess = SpeculativeTestSupport.Uniform(51);
        Func<double> accept = SpeculativeTestSupport.Uniform(52);
        long[] counts = new long[p.Length];
        const int trials = 1_000_000;

        for (int t = 0; t < trials; t++)
        {
            int x = (int)(guess() * p.Length);
            SpeculativeOutcome outcome = RejectionSampler.Verify([x], [p, p], null, accept);
            counts[outcome.Accepted == 1 ? x : outcome.NextToken]++;
        }
        AssertMatches(counts, p);
    }

    [Fact]
    public void A_Draft_Token_With_Zero_Target_Mass_Is_Never_Emitted()
    {
        float[] p = [0.5f, 0.0f, 0.5f];
        Func<double> accept = SpeculativeTestSupport.Uniform(61);
        for (int t = 0; t < 100_000; t++)
        {
            SpeculativeOutcome outcome = RejectionSampler.Verify([1], [p, p], [[0.1f, 0.8f, 0.1f]], accept);
            int emitted = outcome.Accepted == 1 ? 1 : outcome.NextToken;
            Assert.NotEqual(1, emitted);
        }
    }

    [Fact]
    public void Row_Counts_Are_Checked()
    {
        Assert.Throws<ArgumentException>(() => RejectionSampler.Verify([1, 2], [[1f, 0f]], null, () => 0.5));
        Assert.Throws<ArgumentException>(() => RejectionSampler.Verify([1], [[1f, 0f], [1f, 0f]], [[1f, 0f], [1f, 0f]], () => 0.5));
    }

    private static int Inverse(float[] probs, double u)
    {
        double acc = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            acc += probs[i];
            if (u < acc) return i;
        }
        return probs.Length - 1;
    }

    private static void AssertMatches(long[] counts, float[] expected)
    {
        double stat = SpeculativeTestSupport.ChiSquare(counts, expected.Select(v => (double)v).ToArray(), out int df);
        Assert.True(stat < SpeculativeTestSupport.Critical(df), $"chi-square {stat:F2} over {df} df, counts {string.Join(",", counts)}");
    }
}
