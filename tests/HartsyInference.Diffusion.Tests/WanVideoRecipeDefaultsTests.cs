using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Variants;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan's defaults follow the compat class the checkpoint resolved under, as SwarmUI core's Wan branches do:
/// Wan 2.1 runs CFG 6, Wan 2.2 5B 22 fps, and every Wan 81 frames.</summary>
public sealed class WanVideoRecipeDefaultsTests
{
    [Theory]
    [InlineData("wan", 5.0f, 24)]
    [InlineData(WanVideoRecipe.Wan21_14BCompatClassId, 6.0f, 24)]
    [InlineData(WanVideoRecipe.Wan21_1_3BCompatClassId, 6.0f, 24)]
    [InlineData(WanVideoRecipe.Wan22_5BCompatClassId, 5.0f, 22)]
    public void Defaults_FollowTheCompatClass(string familyId, float cfg, int fps)
    {
        VideoDefaults defaults = new WanVideoRecipe(familyId).Defaults;

        Assert.Equal((50, cfg, 832, 480, 81, fps), (defaults.Steps, defaults.CfgScale, defaults.Width, defaults.Height, defaults.Frames, defaults.Fps));
    }

    [Fact]
    public void DefaultsFor_Ti2V5BUnderTheGenericSlug_TakesThe5BDefaults()
    {
        ResolvedModelVariant ti2v = new ResolvedModelVariant(WanVideoVariants.Catalog.FamilyId, WanVideoVariants.Ti2V5B, ModelVariantSource.Structure, "test");

        Assert.Equal(WanVideoRecipe.Wan22_5BDefaults, new WanVideoRecipe().DefaultsFor(ti2v));
    }
}
