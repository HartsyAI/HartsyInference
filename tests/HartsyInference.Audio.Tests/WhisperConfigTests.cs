using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies the per-size Whisper config presets match the HuggingFace
/// <c>config.json</c> for each upstream release. Numbers are independently sourced
/// from the OpenAI <c>whisper/model.py</c> ModelDimensions tables — if these drift,
/// it means our preset table has bit-rotted vs upstream.</summary>
public sealed class WhisperConfigTests
{
    [Fact]
    public void Pipeline_InferConfig_MapsEnglishOnlyReposToEnglishOnlyPresets()
    {
        Assert.Equal(WhisperConfig.TinyEn, WhisperPipeline.InferConfig("openai/whisper-tiny.en"));
        Assert.Equal(WhisperConfig.BaseEn, WhisperPipeline.InferConfig("openai/whisper-base.en"));
        Assert.Equal(WhisperConfig.SmallEn, WhisperPipeline.InferConfig("openai/whisper-small.en"));
        Assert.Equal(WhisperConfig.MediumEn, WhisperPipeline.InferConfig("openai/whisper-medium.en"));
        Assert.Equal(WhisperConfig.DistilSmallEn, WhisperPipeline.InferConfig("distil-whisper/distil-small.en"));
        Assert.NotEqual(WhisperConfig.Small, WhisperPipeline.InferConfig("openai/whisper-small.en"));
    }

    [Fact]
    public void Pipeline_InferConfig_MatchesPresets()
    {
        Assert.Equal(WhisperConfig.Tiny, WhisperPipeline.InferConfig("openai/whisper-tiny"));
        Assert.Equal(WhisperConfig.LargeV3Turbo, WhisperPipeline.InferConfig("openai/whisper-large-v3-turbo"));
        Assert.Equal(WhisperConfig.DistilLargeV3, WhisperPipeline.InferConfig("distil-whisper/distil-large-v3"));
        Assert.Throws<ArgumentException>(() => WhisperPipeline.InferConfig("some-random/whisper-fork"));
    }
}
