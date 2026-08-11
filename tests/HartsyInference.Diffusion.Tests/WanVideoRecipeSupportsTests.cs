using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Regression coverage for the Tier 0.2 <c>wan-22-5b</c>/<c>wan-21-1_3b</c> <see cref="VideoFeatures.EndFrame"/>
/// over-claim fix, <b>updated 2026-08-11 for Tier 3.3's real wiring</b>: <c>wan-22-5b</c> (TI2V-5B) now DOES claim
/// <see cref="VideoFeatures.EndFrame"/> — <see cref="WanVideoRecipePipeline"/>'s non-concat path VAE-encodes
/// <c>VideoRequest.VideoEndFrame</c> into a <c>lastFrameLatent</c> exactly like <c>InitImage</c>'s
/// <c>firstFrameLatent</c>, real-weight verified against the local TI2V-5B checkpoint. <c>wan-21-1_3b</c> shares
/// the identical code path but stays narrowed — no local 1.3B checkpoint exists to verify against, and this
/// backlog's rule is real-checkpoint verification, not "should work by symmetry." Weight-free — <c>Supports</c> is
/// resolved purely from the family id, no checkpoint touched, which is exactly the kind of silent flag/behavior
/// drift a unit test should catch.</summary>
public sealed class WanVideoRecipeSupportsTests
{
    [Fact]
    public void Supports_UnverifiedNonConcatFamily_DoesNotClaimEndFrame()
    {
        VideoFeatures supports = new WanVideoRecipe(WanVideoRecipe.Wan21_1_3BCompatClassId).Supports;

        Assert.Equal(VideoFeatures.None, supports & VideoFeatures.EndFrame);
        Assert.Equal(VideoFeatures.InitImage, supports & VideoFeatures.InitImage);
    }

    [Fact]
    public void Supports_VerifiedTi2V5B_ClaimsEndFrame()
    {
        VideoFeatures supports = new WanVideoRecipe(WanVideoRecipe.Wan22_5BCompatClassId).Supports;

        Assert.Equal(VideoFeatures.EndFrame, supports & VideoFeatures.EndFrame);
        Assert.Equal(VideoFeatures.InitImage, supports & VideoFeatures.InitImage);
    }

    [Theory]
    [InlineData(WanVideoRecipe.Wan21_14BCompatClassId)]
    [InlineData("wan")]
    public void Supports_AmbiguousFamilies_StillClaimsEndFrame(string familyId)
    {
        VideoFeatures supports = new WanVideoRecipe(familyId).Supports;

        Assert.Equal(VideoFeatures.EndFrame, supports & VideoFeatures.EndFrame);
    }

    [Fact]
    public void SupportsFor_NullCheckpoint_FallsBackToNarrowedSupports_1_3B()
    {
        WanVideoRecipe recipe = new WanVideoRecipe(WanVideoRecipe.Wan21_1_3BCompatClassId);

        Assert.Equal(recipe.Supports, recipe.SupportsFor(null));
        Assert.Equal(VideoFeatures.None, recipe.SupportsFor(null) & VideoFeatures.EndFrame);
    }

    [Fact]
    public void SupportsFor_NullCheckpoint_FallsBackToVerifiedSupports_Ti2V5B()
    {
        WanVideoRecipe recipe = new WanVideoRecipe(WanVideoRecipe.Wan22_5BCompatClassId);

        Assert.Equal(recipe.Supports, recipe.SupportsFor(null));
        Assert.Equal(VideoFeatures.EndFrame, recipe.SupportsFor(null) & VideoFeatures.EndFrame);
    }
}
