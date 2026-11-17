using HartsyInference.LLM.Sampling;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Sampling;

/// <summary>Pins the token stream of the default and the filtered sampler chains on fixed logits, so adding a step to the
/// shared chain (the DRY penalty) cannot change what any existing configuration draws.</summary>
public sealed class SamplerChainGoldenTests(ITestOutputHelper output)
{
    private const int Vocab = 512;
    private const int Steps = 300;

    private static ulong Digest(SamplingOptions options)
    {
        SamplerChain chain = SamplerChain.FromOptions(options);
        Random logitRng = new(1234);
        List<int> history = [];
        ulong hash = 14695981039346656037UL;
        float[] logits = new float[Vocab];
        for (int step = 0; step < Steps; step++)
        {
            for (int i = 0; i < Vocab; i++)
            {
                logits[i] = (float)(logitRng.NextDouble() * 6 - 3);
            }
            // A recurring favourite keeps the repetition penalty and the filters doing real work.
            logits[history.Count > 0 ? history[^1] : 7] += 2.5f;
            int token = chain.Next(logits, history);
            history.Add(token);
            hash = (hash ^ (uint)token) * 1099511628211UL;
        }
        return hash;
    }

    [Theory]
    [InlineData("greedy", true, 1.0f, 0, 1.0f, 0.0f, 1.0f, 0UL)]
    [InlineData("temperature_topk_topp", false, 0.7f, 25, 0.9f, 0.0f, 1.0f, 42UL)]
    [InlineData("minp_repetition", false, 0.8f, 0, 1.0f, 0.05f, 1.15f, 7UL)]
    public void Digest_MatchesPinned(string name, bool greedy, float temperature, int topK, float topP, float minP,
        float repetition, ulong seed)
    {
        SamplingOptions options = new()
        {
            Greedy = greedy, Temperature = temperature, TopK = topK, TopP = topP, MinP = minP,
            RepetitionPenalty = repetition, Seed = seed,
        };
        ulong digest = Digest(options);
        output.WriteLine($"GOLDEN {name}: {digest}");
        Assert.True(Pinned.TryGetValue(name, out ulong expected), "no pinned value yet");
        Assert.Equal(expected, digest);
    }

    // Captured on unmodified origin/main (58f248e00).
    private static readonly Dictionary<string, ulong> Pinned = new()
    {
        ["greedy"] = 8426035428634467365UL,
        ["temperature_topk_topp"] = 4055789098893350885UL,
        ["minp_repetition"] = 18263405048095103762UL,
    };
}
