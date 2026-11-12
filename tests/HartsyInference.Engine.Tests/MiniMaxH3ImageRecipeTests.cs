using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>Public surface of the MiniMax-H3-Image recipe. Unit tests only; they need no weights.
/// TODO(gpu): the Engine test project does not compile on main (CudaRedistProvisionerTests.cs), so these do not run
/// until that is fixed. Refusal tests on the internal pipeline need InternalsVisibleTo or a public seam: TODO.</summary>
public sealed class MiniMaxH3ImageRecipeTests
{
    [Theory]
    [InlineData("minimax-h3-image")]
    [InlineData("MINIMAX-H3-IMAGE")]
    public void MatchesTheImageFamilyId(string familyId)
    {
        Assert.True(new MiniMaxH3ImageRecipe().Matches(familyId));
    }

    [Theory]
    [InlineData("minimax-h3")]
    [InlineData("krea2")]
    public void DoesNotMatchOtherFamilies(string familyId)
    {
        Assert.False(new MiniMaxH3ImageRecipe().Matches(familyId));
    }

    [Fact]
    public void DefaultsAreSixStepsAtGuidanceOneAndTwoKSquare()
    {
        ImageDefaults defaults = new MiniMaxH3ImageRecipe().Defaults;

        Assert.Equal(6, defaults.Steps);
        Assert.Equal(1.0f, defaults.CfgScale);
        Assert.Equal(2048, defaults.Width);
        Assert.Equal(2048, defaults.Height);
    }
}
