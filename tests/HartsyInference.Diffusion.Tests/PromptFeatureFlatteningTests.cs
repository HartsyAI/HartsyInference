using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Features;
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

    /// <summary>The generic refiner runs a DIFFERENT family over the base's pixels, and the two need not agree about
    /// weighting. Preparation is destructive in both directions, so the refiner is given the CALLER's prompt and
    /// resolves it against its own mode rather than inheriting whatever the base was left with.</summary>
    [Fact]
    public void ARefinerThatCannotWeightDoesNotInheritTheBasesEmphasis() =>
        Assert.Equal("an orange cat",
            RefinerStage.PrepareForRefiner("an <weight[1.5]:orange> cat", PromptWeightingMode.None, false));

    /// <summary>The direction the first cut of this missed: an unweighted base had already collapsed the tag, so a
    /// refiner that CAN weight silently received no emphasis at all.</summary>
    [Fact]
    public void ARefinerThatCanWeightGetsTheEmphasisEvenWhenTheBaseCouldNot() =>
        Assert.Equal("an (orange:1.5) cat",
            RefinerStage.PrepareForRefiner("an <weight[1.5]:orange> cat", PromptWeightingMode.CondScale, false));

    /// <summary>A segment's sub-prompt belongs to its own masked denoise. The refiner is a full-canvas pass, so it is
    /// stripped there for the same reason the base pass strips it.</summary>
    [Fact]
    public void ARefinerDoesNotInheritASegmentsSubPrompt() =>
        Assert.DoesNotContain("tabby",
            RefinerStage.PrepareForRefiner("a cat <segment:face> tabby", PromptWeightingMode.None, false)!,
            StringComparison.Ordinal);

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

    /// <summary>A <c>fromto</c> flattens to its step-0 TEXT, which is not always its first branch. It switches at
    /// <c>step &lt; when</c>, so <c>&lt;fromto[0]:a, b&gt;</c> has already switched before step 0 runs — and a family
    /// that cannot schedule, which is most of them, sees only this flattened value. Reading "first branch" as the
    /// step-0 value made <c>&lt;fromto[0]:&gt;</c> a silent no-op everywhere.</summary>
    [Theory]
    [InlineData("<fromto[0]:fox, whale>", "whale")]
    [InlineData("<fromto[0.5]:fox, whale>", "fox")]
    [InlineData("<fromto[99]:fox, whale>", "fox")]
    [InlineData("<alternate:fox, whale>", "fox")]
    public void SchedulingFlattensToItsStepZeroText(string prompt, string expected) =>
        Assert.Equal(expected, PromptFeatureFlattening.Prepare(prompt, PromptWeightingMode.CondScale));

    /// <summary>Declaring <see cref="ImageFeatures.PromptScheduling"/> is what STOPS <c>ImagesService</c> collapsing
    /// <c>&lt;alternate:&gt;</c>/<c>&lt;fromto[N]:&gt;</c>, so the raw tag text reaches the recipe. A recipe that
    /// declares it without building a <c>ScheduledPrompt</c> — and without flattening for its own base ids — hands
    /// the encoder the literal tag as prose, which is exactly the bug the Flux.2 gate caught. That cannot be checked
    /// from here, so the bit is ledgered instead: adding a family to this list is the moment to confirm its pipeline
    /// really consumes a schedule.</summary>
    [Fact]
    public void OnlyFamiliesWhosePipelineConsumesAScheduleDeclareIt()
    {
        string[] expected = ["flux2", "mage-flow", "qwen-image", "sd15", "sdxl"];
        string[] declared = [.. RecipeRegistry.DefaultNames
            .Where(family => (RecipeRegistry.Resolve(family)!.Supports & ImageFeatures.PromptScheduling) != 0)
            .Order(StringComparer.Ordinal)];
        Assert.Equal(expected, declared);
    }
}
