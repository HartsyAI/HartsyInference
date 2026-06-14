using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Moonshine config presets must match the published HuggingFace
/// <c>config.json</c> for each variant — these tests pin the per-size hyperparams.</summary>
public sealed class MoonshineConfigTests
{
    [Fact]
    public void Base_MatchesPublishedConfig()
    {
        MoonshineConfig c = MoonshineConfig.Base;
        Assert.Equal(416, c.HiddenSize);
        Assert.Equal(8, c.EncoderLayers);
        Assert.Equal(8, c.DecoderLayers);
        Assert.Equal(8, c.NumHeads);
        Assert.Equal(52, c.HeadDim);     // 416 / 8
        Assert.Equal(1664, c.IntermediateSize);
        Assert.Equal(32_768, c.VocabSize);
        Assert.Equal(194, c.MaxTextPositions);
        Assert.Equal(10_000f, c.RopeTheta);
        Assert.Equal(0.62f, c.PartialRotaryFactor);
    }

    [Fact]
    public void Tiny_MatchesPublishedConfig()
    {
        MoonshineConfig c = MoonshineConfig.Tiny;
        Assert.Equal(288, c.HiddenSize);
        Assert.Equal(6, c.EncoderLayers);
        Assert.Equal(6, c.DecoderLayers);
        Assert.Equal(8, c.NumHeads);
        Assert.Equal(36, c.HeadDim);     // 288 / 8
        Assert.Equal(1152, c.IntermediateSize);
        Assert.Equal(0.9f, c.PartialRotaryFactor);
    }

    [Fact]
    public void RotaryDim_IsEvenInt_OfHeadDimTimesFactor()
    {
        // base: head_dim=52, partial=0.62 → int(32.24) = 32 (already even)
        Assert.Equal(32, MoonshineConfig.Base.RotaryDim);
        // tiny: head_dim=36, partial=0.9 → int(32.4) = 32 (already even)
        Assert.Equal(32, MoonshineConfig.Tiny.RotaryDim);
    }

    [Fact]
    public void ConvStem_TotalDownsample_Is384()
    {
        Assert.Equal(384, MoonshineConfig.Base.TotalDownsample);
        Assert.Equal(384, MoonshineConfig.Tiny.TotalDownsample);
    }

    [Fact]
    public void Pipeline_InferConfig_MatchesPresets()
    {
        Assert.Equal(MoonshineConfig.Base, MoonshinePipeline.InferConfig("UsefulSensors/moonshine-base"));
        Assert.Equal(MoonshineConfig.Tiny, MoonshinePipeline.InferConfig("UsefulSensors/moonshine-tiny"));
        Assert.Throws<ArgumentException>(() => MoonshinePipeline.InferConfig("some/random-fork"));
    }

    [Fact]
    public void SpecialTokens_AreStandard()
    {
        MoonshineConfig c = MoonshineConfig.Base;
        Assert.Equal(1, c.BosTokenId);
        Assert.Equal(2, c.EosTokenId);
        Assert.Equal(2, c.PadTokenId);  // EOS reused as pad
    }
}
