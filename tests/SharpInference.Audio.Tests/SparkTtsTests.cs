using SharpInference.Audio.Dsp;
using SharpInference.Audio.Models.SparkTts;
using SharpInference.Audio.Sampling;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Spark-TTS's testable surface: the config preset and the shared
/// <see cref="NucleusSampler"/> core (now used by both CosyVoice and Spark-TTS). The LM forward + BiCodec
/// decode need the ~3.95 GB checkpoint.</summary>
public sealed class SparkTtsTests
{
    [Fact]
    public void Config_V0_5B_HasExpectedValues()
    {
        SparkTtsConfig c = SparkTtsConfig.V0_5B;
        Assert.Equal(166_000, c.Llm.VocabSize);
        Assert.Equal(896, c.Llm.HiddenSize);
        Assert.Equal(24, c.Llm.NumHiddenLayers);
        Assert.Equal(8_192, c.SemanticVocab);
        Assert.Equal(4_096, c.GlobalVocab);
        Assert.Equal(32, c.NumGlobalTokens);
        Assert.Equal(16_000, c.SampleRate);
        // Semantic tokens precede global tokens in the extended vocab.
        Assert.True(c.GlobalTokenBase >= c.SemanticTokenBase + c.SemanticVocab);
        // BiCodec global FSQ levels multiply to the global vocab.
        int fsq = 1;
        foreach (int l in c.BiCodec.FsqLevels) fsq *= l;
        Assert.Equal(c.GlobalVocab, fsq);
    }

    [Fact]
    public void NucleusSampler_TopK1_ReturnsArgmax()
    {
        uint rng = DeterministicRng.Seed(1);
        Span<float> logits = [0.1f, 0.2f, 5.0f, 0.3f];
        Assert.Equal(2, NucleusSampler.Draw(logits, 4, temperature: 1f, topK: 1, topP: 1f, ref rng));
    }

    [Fact]
    public void NucleusSampler_IsDeterministic_ForFixedRngState()
    {
        Span<float> logits = [1f, 2f, 1.5f, 0.5f, 3f, 0.2f];
        uint a = DeterministicRng.Seed(42);
        uint b = DeterministicRng.Seed(42);
        for (int i = 0; i < 20; i++)
            Assert.Equal(
                NucleusSampler.Draw(logits, 6, 0.8f, 25, 0.9f, ref a),
                NucleusSampler.Draw(logits, 6, 0.8f, 25, 0.9f, ref b));
    }

    [Fact]
    public void NucleusSampler_RespectsCandidateCount()
    {
        uint rng = DeterministicRng.Seed(7);
        // 8-wide buffer but only the first 4 are candidates; indices 4..7 (huge) must never be drawn.
        Span<float> logits = [1f, 1f, 1f, 1f, 99f, 99f, 99f, 99f];
        for (int i = 0; i < 100; i++)
            Assert.InRange(NucleusSampler.Draw(logits, 4, 1f, 25, 1f, ref rng), 0, 3);
    }

    [Fact]
    public void NucleusSampler_MaskToken_IsNeverDrawn()
    {
        uint rng = DeterministicRng.Seed(3);
        Span<float> logits = [0f, 20f, 0.5f, 0.5f];   // index 1 dominates
        for (int i = 0; i < 50; i++)
            Assert.NotEqual(1, NucleusSampler.Draw(logits, 4, 1f, 5, 1f, ref rng, maskToken: 1));
    }
}
