using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Regression coverage for the <c>wan-22-5b</c>/<c>wan-21-1_3b</c> <see cref="VideoFeatures.EndFrame"/>
/// over-claim fix: both compat classes are always driven by <see cref="WanVideoRecipePipeline"/>'s non-concat
/// path, which never reads <c>VideoRequest.VideoEndFrame</c>, so <see cref="WanVideoRecipe.Supports"/> must not
/// claim the flag for them. Weight-free — <c>Supports</c> is resolved purely from the family id, no checkpoint
/// touched, which is exactly the kind of silent flag/behavior drift a unit test should catch.</summary>
public sealed class WanVideoRecipeSupportsTests
{
    [Theory]
    [InlineData(WanVideoRecipe.Wan22_5BCompatClassId)]
    [InlineData(WanVideoRecipe.Wan21_1_3BCompatClassId)]
    public void Supports_NonConcatOnlyFamilies_DoesNotClaimEndFrame(string familyId)
    {
        VideoFeatures supports = new WanVideoRecipe(familyId).Supports;

        Assert.Equal(VideoFeatures.None, supports & VideoFeatures.EndFrame);
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

    [Theory]
    [InlineData(WanVideoRecipe.Wan22_5BCompatClassId)]
    [InlineData(WanVideoRecipe.Wan21_1_3BCompatClassId)]
    public void SupportsFor_NullCheckpoint_FallsBackToNarrowedSupports(string familyId)
    {
        WanVideoRecipe recipe = new WanVideoRecipe(familyId);

        Assert.Equal(recipe.Supports, recipe.SupportsFor(null));
        Assert.Equal(VideoFeatures.None, recipe.SupportsFor(null) & VideoFeatures.EndFrame);
    }
}
