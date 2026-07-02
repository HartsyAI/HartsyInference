using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Sesame CSM's config + the Qwen2→Llama reuse generalization.</summary>
public sealed class CsmTests
{
    [Fact]
    public void Config_V1B_DualTransformer_HasExpectedShape()
    {
        CsmConfig c = CsmConfig.V1B;
        // Backbone = Llama-3.2-1B headless body.
        Assert.Equal(2_048, c.Backbone.HiddenSize);
        Assert.Equal(16, c.Backbone.NumHiddenLayers);
        Assert.Equal(32, c.Backbone.NumAttentionHeads);
        Assert.Equal(8, c.Backbone.NumKeyValueHeads);
        Assert.False(c.Backbone.AttentionBias);          // Llama = no Q/K/V bias
        // Decoder = Llama-100M.
        Assert.Equal(1_024, c.Decoder.HiddenSize);
        Assert.Equal(4, c.Decoder.NumHiddenLayers);
        Assert.False(c.Decoder.AttentionBias);
        Assert.Equal(32, c.NumCodebooks);                // 32 Mimi codebooks (decoder MaxPositionEmbeddings = 33 = NumCodebooks + 1)
        Assert.Equal(24_000, c.SampleRate);
        Assert.Equal(1_920, c.FrameSamples);             // 80 ms @ 24 kHz
    }

    [Fact]
    public void Qwen2Config_AttentionBias_DefaultsTrue_PreservesExistingModels()
    {
        // The bias generalization must not change Qwen2.5 behavior (CosyVoice / SparkTTS / VibeVoice).
        Assert.True(Qwen2Config.Qwen25_0_5B.AttentionBias);
        Assert.True(Qwen2Config.Qwen25_1_5B.AttentionBias);
        Assert.True(Qwen2Config.Qwen25_7B.AttentionBias);
    }
}
