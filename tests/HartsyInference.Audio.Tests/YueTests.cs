using HartsyInference.Audio.Models.Music;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free config test for YuE — confirms the Stage-1 LM is set up as a LLaMA-2-7B body
/// reusing the (bias-off, MHA) Qwen2 transformer.</summary>
public sealed class YueTests
{
    [Fact]
    public void Config_V1_Stage1_IsLlama2_7B()
    {
        YueConfig c = YueConfig.V1;
        Assert.Equal(4_096, c.Stage1.HiddenSize);
        Assert.Equal(32, c.Stage1.NumHiddenLayers);
        Assert.Equal(32, c.Stage1.NumAttentionHeads);
        Assert.Equal(32, c.Stage1.NumKeyValueHeads);     // MHA (no GQA) for Llama-2-7B
        Assert.False(c.Stage1.AttentionBias);            // Llama = no bias
        Assert.Equal(11_008, c.Stage1.IntermediateSize);
        Assert.Equal(8, c.NumCodebooks);
        Assert.Equal(16_000, c.SampleRate);
        Assert.Equal(50, c.FrameRateHz);
        Assert.Equal(1.1f, c.RepetitionPenalty);         // mandatory per YuE
        // Accompaniment track base follows the vocal track by one codebook.
        Assert.Equal(c.VocalTokenBase + c.CodebookSize, c.AccompTokenBase);
    }
}
