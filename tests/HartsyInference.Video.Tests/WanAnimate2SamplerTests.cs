using Xunit;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>Which solver an unspecified request lands on. Silent when wrong: UniPC and FlowDPM++ 2M both produce a
/// plausible clip, so a default that quietly reverted to UniPC would only show up in the log line.</summary>
public sealed class WanAnimate2SamplerTests
{
    [Fact]
    public void ResolveSampler_DefaultsToTheReferenceSolver_AndKeepsUniPcSelectable()
    {
        Assert.Equal(WanAnimate2Pipeline.DefaultSampler, WanAnimate2Pipeline.ResolveSampler(null));
        Assert.Equal(WanAnimate2Pipeline.DefaultSampler, WanAnimate2Pipeline.ResolveSampler("  "));
        Assert.NotEqual(WanAnimate2Pipeline.AlternateSampler, WanAnimate2Pipeline.ResolveSampler(null));

        Assert.Equal(WanAnimate2Pipeline.DefaultSampler, WanAnimate2Pipeline.ResolveSampler("DPM++2M"));
        Assert.Equal(WanAnimate2Pipeline.AlternateSampler, WanAnimate2Pipeline.ResolveSampler("UniPC"));

        Assert.Throws<NotSupportedException>(() => WanAnimate2Pipeline.ResolveSampler("euler"));
    }
}
