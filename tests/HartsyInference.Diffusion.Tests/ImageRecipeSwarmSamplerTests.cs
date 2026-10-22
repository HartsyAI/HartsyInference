using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the samplers taken from SwarmUI core (RECIPE_CONTRACT.md ledger): the full Anima and Flux 3 Image builds
/// default to <c>er_sde</c>, while the distilled builds keep their pipeline's own sampler.</summary>
public sealed class ImageRecipeSwarmSamplerTests
{
    [Fact]
    public void FullBuilds_DefaultToErSde_DistilledBuildsLeaveTheSamplerToThePipeline()
    {
        Assert.Equal(("er_sde", 2), (AnimaRecipe.FamilyDefaults.Sampler, AnimaRecipe.FamilyDefaults.Revision));
        Assert.Equal(("er_sde", 2), (Flux3Recipe.FamilyDefaults.Sampler, Flux3Recipe.FamilyDefaults.Revision));
        Assert.Null(AnimaRecipe.TurboDefaults.Sampler);
        Assert.Null(Flux3Recipe.DistilledDefaults.Sampler);
    }
}
