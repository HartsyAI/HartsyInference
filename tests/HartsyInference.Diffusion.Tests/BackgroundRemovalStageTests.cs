using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Gating of <see cref="BackgroundRemovalStage"/>. Weight-free: the only thing that can silently go wrong
/// without a GPU is the flag test itself — a stage that declines to run leaves no trace in the result, which is
/// exactly how <c>removebackground</c> used to be accepted and dropped. A null engine stands in for "no model was
/// resolved and no net was loaded": reaching for either would throw.</summary>
public sealed class BackgroundRemovalStageTests
{
    private static readonly ImageResult Result = new ImageResult { Rgb = [1, 2, 3], Width = 1, Height = 1 };

    [Fact]
    public void Apply_WithoutTheFlag_ReturnsTheResultUntouched()
    {
        ImageRequest request = new ImageRequest { Prompt = "a cat" };
        Assert.Same(Result, BackgroundRemovalStage.Apply(null!, request, Result));
    }

    [Fact]
    public void Apply_WithTheFlagOff_ReturnsTheResultUntouched()
    {
        ImageRequest request = new ImageRequest { Prompt = "a cat", RemoveBackground = false };
        Assert.Same(Result, BackgroundRemovalStage.Apply(null!, request, Result));
    }

    [Fact]
    public void Apply_WithTheFlagOn_RunsTheMatte()
    {
        ImageRequest request = new ImageRequest { Prompt = "a cat", RemoveBackground = true };
        Assert.ThrowsAny<Exception>(() => BackgroundRemovalStage.Apply(null!, request, Result));
    }

    [Fact]
    public void MissizedAlpha_IsNotAnAlphaChannel()
    {
        // Guards the encode side of the contract: PngEncoder asks HasAlpha, so a plane that does not cover the
        // image must fall back to color type 2 rather than produce a corrupt RGBA PNG.
        Assert.False((Result with { Alpha = [255, 255] }).HasAlpha);
        Assert.True((Result with { Alpha = [255] }).HasAlpha);
    }
}
