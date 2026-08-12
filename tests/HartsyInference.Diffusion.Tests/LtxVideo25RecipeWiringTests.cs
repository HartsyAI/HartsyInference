using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Checks that both LTX-2.5 ids reach a recipe and that the distilled one carries its own sampling
/// contract. The distilled and dev checkpoints are indistinguishable, so the id is the only carrier of that
/// intent — a missing registration would silently sample the distilled model on the dev schedule.</summary>
public sealed class LtxVideo25RecipeWiringTests
{
    [Theory]
    [InlineData("ltx-2.5")]
    [InlineData("ltx-2.5-distilled")]
    [InlineData("ltx-2")]
    [InlineData("ltx-2.3")]
    [InlineData("lightricks-ltx-video-2")]
    public void FamilyIdResolvesToARecipe(string familyId)
    {
        Assert.NotNull(VideoRecipeRegistry.Resolve(familyId));
    }

    [Fact]
    public void DistilledAndDevResolveToDifferentRecipes()
    {
        IVideoRecipe dev = VideoRecipeRegistry.Resolve("ltx-2.5")!;
        IVideoRecipe distilled = VideoRecipeRegistry.Resolve("ltx-2.5-distilled")!;

        Assert.NotEqual(dev.Name, distilled.Name);
        Assert.Equal(LtxVideo2Config.V25Distilled.NumInferenceSteps, distilled.Defaults.Steps);
        Assert.Equal(LtxVideo2Config.V25Distilled.GuidanceScale, distilled.Defaults.CfgScale);
        Assert.Equal(50, dev.Defaults.Steps);
        Assert.Equal(3.0f, dev.Defaults.CfgScale);
    }

    [Fact]
    public void DistilledRecipeDoesNotClaimTheOlderFamilies()
    {
        LtxVideo2Recipe distilled = new LtxVideo2Recipe(distilled: true);

        Assert.True(distilled.Matches("ltx-2.5-distilled"));
        Assert.False(distilled.Matches("ltx-2.5"));
        Assert.False(distilled.Matches("ltx-2"));
        Assert.False(distilled.Matches("ltx-2.3"));
    }

    [Theory]
    [InlineData("ltx-2.5")]
    [InlineData("ltx-2.5-distilled")]
    public void CatalogExposesTheIdToTheCli(string id)
    {
        CatalogEntry? entry = ModelCatalog.All.FirstOrDefault(e => e.Id == id);

        Assert.NotNull(entry);
        Assert.True(entry!.CliDrivable, $"'{id}' must be CLI-drivable to be reachable via hartsy video -m {id}");
    }
}
