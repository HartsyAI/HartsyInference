using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Moonshine config presets must match the published HuggingFace
/// <c>config.json</c> for each variant — these tests pin the per-size hyperparams.</summary>
public sealed class MoonshineConfigTests
{
    [Fact]
    public void Pipeline_InferConfig_MatchesPresets()
    {
        Assert.Equal(MoonshineConfig.Base, MoonshinePipeline.InferConfig("UsefulSensors/moonshine-base"));
        Assert.Equal(MoonshineConfig.Tiny, MoonshinePipeline.InferConfig("UsefulSensors/moonshine-tiny"));
        Assert.Throws<ArgumentException>(() => MoonshinePipeline.InferConfig("some/random-fork"));
    }

}
