using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.FishAudio;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The Fish Audio top-k / top-p draw: the highest-ranked candidate always survives, a lower one only while the inclusive
/// cumulative stays within top-p, and S2 and Drama 3 differ only in whether that cumulative reads the tempered distribution.</summary>
public sealed class FishAudioSamplingTests
{
    private static readonly float[] Logits = [MathF.Log(0.6f), MathF.Log(0.3f), MathF.Log(0.1f)];

    [Fact]
    public void TheTokenCrossingTopP_IsDropped()
    {
        FishAudioSampling.Scratch scratch = new(3);
        uint rng = DeterministicRng.Seed(1);
        for (int i = 0; i < 200; i++)
        {
            Assert.Equal(0, FishAudioSampling.Sample(Logits, 1f, 0.7f, 3, ref rng, scratch, temperatureFirst: true));
        }
    }

    [Fact]
    public void Drama3FiltersTheTemperedDistribution_AndS2TheRawOne()
    {
        // At temperature 2 the tempered cumulative after two tokens is 0.81, under top-p 0.85; the raw one is 0.9, over it.
        FishAudioSampling.Scratch scratch = new(3);
        uint rng = DeterministicRng.Seed(2);
        HashSet<int> drama = [], s2 = [];
        for (int i = 0; i < 400; i++)
        {
            drama.Add(FishAudioSampling.Sample(Logits, 2f, 0.85f, 3, ref rng, scratch, temperatureFirst: true));
            s2.Add(FishAudioSampling.Sample(Logits, 2f, 0.85f, 3, ref rng, scratch, temperatureFirst: false));
        }
        Assert.Equal([0, 1], drama.Order());
        Assert.Equal([0], s2);
    }

    [Fact]
    public void WideTopK_RanksLikeANarrowOne()
    {
        float[] logits = new float[600];
        Random random = new(3);
        for (int i = 0; i < logits.Length; i++) logits[i] = random.NextSingle() * 4;
        FishAudioSampling.Scratch scratch = new(logits.Length);
        uint narrowRng = DeterministicRng.Seed(4), wideRng = DeterministicRng.Seed(4);
        for (int i = 0; i < 50; i++)
        {
            // top-p 0.05 keeps only the head of the ranking, so a 300-wide top-k must pick exactly what a 30-wide one does.
            int narrow = FishAudioSampling.Sample(logits, 1f, 0.05f, 30, ref narrowRng, scratch, temperatureFirst: true);
            int wide = FishAudioSampling.Sample(logits, 1f, 0.05f, 300, ref wideRng, scratch, temperatureFirst: true);
            Assert.Equal(narrow, wide);
        }
    }

    [Fact]
    public void Drama3RedrawsARepeatedAudioToken_ButNeverEos()
    {
        FishAudioDrama3Config config = FishAudioDrama3Config.FromJson(
            File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioDrama3", "config.json")));
        int[] candidates = [2, 40, 41, 42, 43, 44, 45, 46, 47];
        // At temperature 0.1 candidate 41 wins every first draw; the redraw (temperature 1, top-p 0.95) reaches the others too.
        float[] logits = [0, 0, 6, 4.5f, 4.5f, 4.5f, 4.5f, 4.5f, 4.5f];
        FishAudioDrama3Sampler sampler = new(config, seed: 9, temperature: 0.1f, topP: 1f, topK: 30);
        HashSet<int> fresh = [], repeated = [];
        for (int i = 0; i < 40; i++)
        {
            sampler.Prime([3]);
            fresh.Add(sampler.SampleMain(logits, candidates));
            sampler.Prime([41]);
            repeated.Add(sampler.SampleMain(logits, candidates));
        }
        Assert.Equal([41], fresh);
        Assert.True(repeated.Count > 1);

        float[] eosLogits = [8, 0, 0, 0, 0, 0, 0, 0, 0];
        sampler.Prime([2]);
        Assert.Equal(2, sampler.SampleMain(eosLogits, candidates));
    }
}
