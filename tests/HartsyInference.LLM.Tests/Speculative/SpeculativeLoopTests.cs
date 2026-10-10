using HartsyInference.LLM.Generation.Speculative;
using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculative;

public sealed class SpeculativeLoopTests
{
    private const int Vocab = 6;

    /// <summary>A stateless target: its logits depend only on the last two tokens, so repeated text gives repeated distributions.</summary>
    private sealed class ToyScorer : ISpeculativeScorer
    {
        public void Score(ReadOnlySpan<int> context, ReadOnlySpan<int> draft, float[][] rows)
        {
            int[] seq = [.. context, .. draft];
            for (int j = 0; j < rows.Length; j++)
            {
                int end = context.Length + j;
                Logits(end >= 2 ? seq[end - 2] : 0, end >= 1 ? seq[end - 1] : 0, rows[j]);
            }
        }

        public static void Logits(int a, int b, float[] row)
        {
            for (int t = 0; t < row.Length; t++) row[t] = (float)(Math.Sin(a * 13.0 + b * 7.0 + t * 2.1) * 2.2);
        }
    }

    private static int PlainNext(SamplerChain chain, List<int> tokens)
    {
        float[] logits = new float[Vocab];
        ToyScorer.Logits(tokens.Count >= 2 ? tokens[^2] : 0, tokens.Count >= 1 ? tokens[^1] : 0, logits);
        return chain.Next(logits, tokens);
    }

    [Fact]
    public void Greedy_Speculation_Reproduces_Plain_Greedy_Decoding_Token_For_Token()
    {
        SamplingOptions greedy = new() { Greedy = true };
        List<int> plain = [0, 1];
        SamplerChain plainChain = SamplerChain.FromOptions(greedy);
        for (int i = 0; i < 40; i++) plain.Add(PlainNext(plainChain, plain));

        List<int> speculative = [0, 1];
        int produced = SpeculativeLoop.Generate(new ToyScorer(), new PromptLookupProposer(), SamplerChain.FromOptions(greedy),
            SpeculativeTestSupport.Uniform(1), speculative, 40, 4, Vocab);
        Assert.Equal(40, produced);
        Assert.Equal(plain, speculative);
    }

    [Fact]
    public void First_Token_Of_A_Drafted_Round_Follows_The_Plain_Sampler()
    {
        // the context ends in a repeated bigram, so the lookup proposer drafts, and the loop's first new token must still follow the target
        SamplingOptions options = new() { Temperature = 0.9f, TopK = 4, Seed = 23 };
        const int trials = 100_000;
        long[] speculative = new long[Vocab], plain = new long[Vocab];
        Func<double> uniform = SpeculativeTestSupport.Uniform(77);
        SamplerChain plainChain = SamplerChain.FromOptions(options);
        for (int t = 0; t < trials; t++)
        {
            List<int> tokens = [0, 1, 2, 0, 1];
            SpeculativeLoop.Generate(new ToyScorer(), new PromptLookupProposer(), SamplerChain.FromOptions(options), uniform, tokens, 3, 3, Vocab);
            speculative[tokens[5]]++;
            plain[PlainNext(plainChain, [0, 1, 2, 0, 1])]++;
        }
        double stat = SpeculativeTestSupport.TwoSampleChiSquare(speculative, plain, out int df);
        Assert.True(stat < SpeculativeTestSupport.Critical(df), $"two-sample chi-square {stat:F2} over {df} df: spec {string.Join(",", speculative)} plain {string.Join(",", plain)}");
    }
}
