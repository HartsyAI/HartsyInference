using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the two ways the distilled registration can go wrong quietly. An id that fails to resolve at
/// all is loud and needs no test; an id that resolves to the wrong sampling contract is not.</summary>
public sealed class LtxVideo25RecipeWiringTests
{
    [Fact]
    public void DistilledRecipeDoesNotClaimTheOlderFamilies()
    {
        // The worst case in the whole registration: if the distilled recipe matched the 2.3 family ids, every
        // existing LTX-2 user would silently be resampled onto an 8-step schedule their checkpoint never saw.
        LtxVideo2Recipe distilled = new LtxVideo2Recipe(distilled: true);

        Assert.True(distilled.Matches("ltx-2.5-distilled"));
        Assert.False(distilled.Matches("ltx-2.5"));
        Assert.False(distilled.Matches("ltx-2"));
        Assert.False(distilled.Matches("ltx-2.3"));
        Assert.False(distilled.Matches("lightricks-ltx-video-2"));
    }

    [Fact]
    public void DistilledAndDevCarryDifferentSamplingDefaults()
    {
        IVideoRecipe dev = VideoRecipeRegistry.Resolve("ltx-2.5")!;
        IVideoRecipe distilled = VideoRecipeRegistry.Resolve("ltx-2.5-distilled")!;

        Assert.Equal(LtxVideo2Config.V25Distilled.NumInferenceSteps, distilled.Defaults.Steps);
        Assert.Equal(LtxVideo2Config.V25Distilled.GuidanceScale, distilled.Defaults.CfgScale);
        Assert.Equal(50, dev.Defaults.Steps);
        Assert.Equal(3.0f, dev.Defaults.CfgScale);
    }
}
