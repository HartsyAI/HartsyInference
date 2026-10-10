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

    /// <summary>The two Animate-2 builds want incompatible settings and a checkpoint announces which it is only
    /// through its file name, so running the base weights at the distillation build's 6 steps / cfg 1 was silent.
    /// It renders hazy, and the haze grows with the frame count — which is why it was read as the driving video
    /// being ignored at long clips rather than as under-denoising (see WAN_ANIMATE2_PARITY_PLAN.md).</summary>
    [Theory]
    [InlineData(false, 6, 1f, true)]      // the settings that produced the mush
    [InlineData(false, 40, 3f, false)]    // upstream's base config
    [InlineData(true, 6, 1f, false)]      // upstream's distillation config
    public void SettingsMismatchWarning_FiresWhenABuildIsSampledLikeTheOther(bool distill, int steps, float guidance, bool expectWarning)
    {
        string? warning = WanAnimate2Pipeline.SettingsMismatchWarning(distill, steps, guidance);
        Assert.Equal(expectWarning, warning is not null);
    }
}
