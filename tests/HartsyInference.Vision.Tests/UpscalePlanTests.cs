using HartsyInference.Vision.Upscale;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Pass-count and final-size arithmetic for target-fitted upscales. A wrong plan is silent: the image
/// still comes back, just stretched by interpolation instead of super-resolved, or four times larger than asked.</summary>
public sealed class UpscalePlanTests
{
    [Fact]
    public void NoTarget_IsOnePassAtModelFactor()
    {
        UpscalePlan plan = UpscalePlan.Create(256, 192, 4, null, null);
        Assert.Equal(new UpscalePlan(1, 1024, 768), plan);
    }

    [Fact]
    public void TargetInsideOnePass_ResizesDown()
    {
        UpscalePlan plan = UpscalePlan.Create(256, 256, 4, 768, 768);
        Assert.Equal(new UpscalePlan(1, 768, 768), plan);
    }

    [Fact]
    public void TargetBeyondOnePass_ChainsASecondPass()
    {
        // 256 → 512 → 1024 on x2plus: two passes cover the target exactly.
        UpscalePlan plan = UpscalePlan.Create(256, 256, 2, 1024, 1024);
        Assert.Equal(new UpscalePlan(2, 1024, 1024), plan);
    }

    [Fact]
    public void TargetBeyondPassCap_ClampsToWhatThePassesProduce()
    {
        // x2 twice is 1024; asking for 4096 must not stretch past it.
        UpscalePlan plan = UpscalePlan.Create(256, 256, 2, 4096, 4096);
        Assert.Equal(new UpscalePlan(2, 1024, 1024), plan);
    }

    [Fact]
    public void OneSideOnly_DerivesTheOtherFromAspect()
    {
        UpscalePlan plan = UpscalePlan.Create(200, 100, 4, 600, null);
        Assert.Equal(new UpscalePlan(1, 600, 300), plan);
    }

    [Fact]
    public void TargetSmallerThanSource_StillRunsOnePassThenDownsizes()
    {
        UpscalePlan plan = UpscalePlan.Create(512, 512, 4, 256, 256);
        Assert.Equal(new UpscalePlan(1, 256, 256), plan);
    }

    [Fact]
    public void InvalidInputs_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => UpscalePlan.Create(0, 10, 4, null, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => UpscalePlan.Create(10, 10, 1, null, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => UpscalePlan.Create(10, 10, 4, 0, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => UpscalePlan.Create(10, 10, 4, 10, 10, maxPasses: 0));
    }
}
