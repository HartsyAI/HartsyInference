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
    public void BuildNoise_IsSeededAndStandardNormal()
    {
        using Tensor a = AukPipeline.BuildNoise(500, 64, 7);
        using Tensor b = AukPipeline.BuildNoise(500, 64, 7);
        using Tensor c = AukPipeline.BuildNoise(500, 64, 8);
        Assert.Equal([1, 500, 64], [a.Shape[0], a.Shape[1], a.Shape[2]]);
        Assert.True(a.AsSpan<float>().SequenceEqual(b.AsSpan<float>()));
        Assert.False(a.AsSpan<float>().SequenceEqual(c.AsSpan<float>()));
        double sum = 0;
        double sq = 0;
        foreach (float v in a.AsSpan<float>())
        {
            sum += v;
            sq += (double)v * v;
        }
        int n = 500 * 64;
        Assert.InRange(sum / n, -0.05, 0.05);
        Assert.InRange(sq / n, 0.95, 1.05);
    }

    [Fact]
    public void FlashSchedule_IgnoresStepAndGuidanceOverrides()
    {
        AukOptions overrides = new() { Steps = 50, CfgScale = 5f, SwayCoef = 0.3f };
        AukPipeline flash = new("flash", true, _ => [1], new AukConfig(), null, null, null);
        AukSchedule s = flash.BuildSchedule(overrides);
        Assert.Equal(4, s.Steps);
        Assert.False(s.UsesCfg);
        Assert.Equal(AukSchedule.FlashGrid.ToArray(), s.Timesteps);

        AukPipeline baseModel = new("base", false, _ => [1], new AukConfig(), null, null, null);
        AukSchedule b = baseModel.BuildSchedule(overrides);
        Assert.Equal(50, b.Steps);
        Assert.Equal(5f, b.Cfg);
        Assert.Equal(32, baseModel.BuildSchedule(new AukOptions()).Steps);
        Assert.Equal(2f, baseModel.BuildSchedule(new AukOptions()).Cfg);
    }

    [Fact]
    public void Constructor_RejectsMismatchedComponentWidths() =>
        Assert.Throws<ArgumentException>(() => new AukPipeline("x", true, _ => [1], new AukConfig { TextDim = 1_024 }));
}
