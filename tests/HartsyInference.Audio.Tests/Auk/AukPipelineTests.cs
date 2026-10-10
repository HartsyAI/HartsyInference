using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Orchestration math of <see cref="AukPipeline"/> that needs no weights: reference trimming, the 30 s budget, noise seeding and the Flash schedule pin.</summary>
public sealed class AukPipelineTests
{
    [Fact]
    public void PrepareReference_TrimsToWholeHops_AtTheTargetRate()
    {
        float[] audio = new float[24_000];
        Assert.Equal(24_000, AukPipeline.PrepareReference(audio, 24_000, 24_000, 480).Length);
        Assert.Equal(23_520, AukPipeline.PrepareReference(new float[23_999], 24_000, 24_000, 480).Length);
        // 1 s at 16 kHz resamples to 24 000 samples (50 whole hops); 1.001 s drops the partial hop.
        Assert.Equal(24_000, AukPipeline.PrepareReference(new float[16_000], 16_000, 24_000, 480).Length);
        Assert.Equal(24_000, AukPipeline.PrepareReference(new float[16_016], 16_000, 24_000, 480).Length);
    }

    [Fact]
    public void PrepareReference_ShorterThanOneFrame_Throws() =>
        Assert.Throws<ArgumentException>(() => AukPipeline.PrepareReference(new float[479], 24_000, 24_000, 480));

    [Fact]
    public void ResolveFrames_EnforcesTheThirtySecondBudget()
    {
        Assert.Equal(1_500, AukPipeline.ResolveFrames(30.0, 24_000, 480));
        Assert.Equal(1, AukPipeline.ResolveFrames(0.001, 24_000, 480));
        Assert.Throws<ArgumentException>(() => AukPipeline.ResolveFrames(30.01, 24_000, 480));
    }

    [Fact]
    public void Constructor_RejectsMismatchedComponentWidths() =>
        Assert.Throws<ArgumentException>(() => new AukPipeline("x", true, _ => [1], new AukConfig { TextDim = 1_024 }));
}
