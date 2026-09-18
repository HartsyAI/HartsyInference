using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the routing both services share: which of SwarmUI's literal prompt tags survive to a recipe is decided
/// by that recipe's declared <see cref="PromptWeightingMode"/>, and by nothing else. Getting it backwards is silent —
/// a family that cannot weight receives <c>(cat:1.5)</c> and encodes the parens and digits as prose.</summary>
public sealed class PromptFeatureFlatteningTests
{
    [Theory]
    [InlineData(PromptWeightingMode.ComfyBlend)]
    [InlineData(PromptWeightingMode.CondScale)]
    [InlineData(PromptWeightingMode.CondScaleWithAttention)]
    public void ADeclaredModeKeepsTheEmphasisGrammar(PromptWeightingMode mode) =>
        Assert.Equal("an (orange:1.5) cat", PromptFeatureFlattening.Prepare("an <weight[1.5]:orange> cat", mode));

    [Fact]
    public void NoDeclaredModeCollapsesTheTagToItsInnerText() =>
        Assert.Equal("an orange cat",
            PromptFeatureFlattening.Prepare("an <weight[1.5]:orange> cat", PromptWeightingMode.None));

    [Fact]
    public void SchedulingTagsCollapseToTheirStepZeroValueByDefault() =>
        Assert.Equal("a cat", PromptFeatureFlattening.Prepare("a <fromto[0.5]:cat, dog>", PromptWeightingMode.None));

    [Fact]
    public void SchedulingTagsSurviveForARecipeThatResolvesThemPerStep() =>
        Assert.Equal("a <fromto[0.5]:cat, dog>",
            PromptFeatureFlattening.Prepare("a <fromto[0.5]:cat, dog>", PromptWeightingMode.ComfyBlend, true));

    [Fact]
    public void ANullPromptFlattensToEmptyRatherThanThrowing() =>
        Assert.Equal("", PromptFeatureFlattening.Prepare(null, PromptWeightingMode.ComfyBlend));

    /// <summary>The generic refiner runs a DIFFERENT family over the base's pixels with the base's prompt. A refiner
    /// that cannot weight must not inherit the base's emphasis parens, and by then the tags are gone — so it is the
    /// parens themselves that have to be resolved, which tag flattening alone would not touch.</summary>
    [Fact]
    public void RebindingForAnUnweightedRefinerStripsTheBasesEmphasisParens() =>
        Assert.Equal("an orange cat",
            PromptFeatureFlattening.Rebind("an (orange:1.5) cat", PromptWeightingMode.None));

    [Fact]
    public void RebindingForAWeightingRefinerKeepsTheEmphasisIntact() =>
        Assert.Equal("an (orange:1.5) cat",
            PromptFeatureFlattening.Rebind("an (orange:1.5) cat", PromptWeightingMode.ComfyBlend));

    /// <summary>The feature bit is derived from the mode in one place, so no recipe may set it by hand — a recipe that
    /// did would keep the emphasis parens in its prompt without anything downstream acting on them.</summary>
    [Fact]
    public void NoRecipeDeclaresThePromptWeightingBitDirectly()
    {
        foreach (string family in RecipeRegistry.DefaultNames)
        {
            IArchitectureRecipe recipe = RecipeRegistry.Resolve(family)!;
            Assert.True((recipe.Supports & ImageFeatures.PromptWeighting) == 0,
                $"'{family}' sets ImageFeatures.PromptWeighting directly; declare PromptWeighting instead.");
        }
        foreach (string family in VideoRecipeRegistry.DefaultNames)
        {
            IVideoRecipe recipe = VideoRecipeRegistry.Resolve(family)!;
            Assert.True((recipe.Supports & VideoFeatures.PromptWeighting) == 0,
                $"'{family}' sets VideoFeatures.PromptWeighting directly; declare PromptWeighting instead.");
        }
    }
}
