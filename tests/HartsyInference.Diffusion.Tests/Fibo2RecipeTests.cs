using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 recipe declarations: the family ids it answers to, its variant split, its defaults, and what it claims to support.</summary>
public sealed class Fibo2RecipeTests
{
    [Theory]
    [InlineData("fibo2")]
    [InlineData("FIBO2")]
    [InlineData("fibo2-dev")]
    public void Matches_FamilyAndCatalogIds(string familyId)
    {
        Assert.True(new Fibo2Recipe().Matches(familyId));
    }

    [Theory]
    [InlineData("fibo")]
    [InlineData("fibo2-edit")]
    [InlineData("flux2")]
    public void Matches_RejectsOtherFamilies(string familyId)
    {
        Assert.False(new Fibo2Recipe().Matches(familyId));
    }

    [Fact]
    public void Registry_ResolvesFiboToItsRecipe()
    {
        Assert.Contains("fibo2", RecipeRegistry.RegisteredNames);
        Assert.IsType<Fibo2Recipe>(RecipeRegistry.Resolve("fibo2-dev"));
    }

    [Fact]
    public void Supports_DeclaresEveryImplementedFeature()
    {
        Fibo2Recipe recipe = new();

        Assert.Equal(ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Lora | ImageFeatures.RefEdit, recipe.Supports);
        Assert.Equal(recipe.Supports, ((IArchitectureRecipe)recipe).SupportsFor(null));
    }

    [Fact]
    public void InputLimits_AllowFourImagesWithoutAnInitImage()
    {
        ImageInputLimits limits = new Fibo2Recipe().InputLimits;

        Assert.Equal(4, limits.MaxImages);
        Assert.Equal(4, limits.MaxReferences);
        Assert.False(limits.ReferencesRequireInitImage);
    }

    [Fact]
    public void FamilyDefaults_AreTheDevSettings()
    {
        ImageDefaults defaults = new Fibo2Recipe().Defaults;

        Assert.Equal(50, defaults.Steps);
        Assert.Equal(5.0f, defaults.CfgScale);
        Assert.Equal(1024, defaults.Width);
        Assert.Equal(1024, defaults.Height);
        Assert.Equal(Math.Exp(1.15), defaults.SigmaShift);
        Assert.Null(defaults.Sampler);
        Assert.Null(defaults.Scheduler);
    }

    [Fact]
    public void EditTurboDefaults_AreTheWorkflowSettings()
    {
        ImageDefaults defaults = Fibo2Recipe.EditTurboDefaults;

        Assert.Equal(4, defaults.Steps);
        Assert.Equal(1.0f, defaults.CfgScale);
        Assert.Equal(1024, defaults.Width);
        Assert.Equal(704, defaults.Height);
        Assert.Equal(Math.Exp(1.15), defaults.SigmaShift);
    }

    [Fact]
    public void Variants_SplitDevFromEditTurboByFileName()
    {
        Fibo2Recipe recipe = new();

        ResolvedModelVariant dev = ModelVariantResolver.Classify(Fibo2Variants.Catalog, FileNameProbe("fibo2_dev_bf16"), []);
        ResolvedModelVariant editTurbo = ModelVariantResolver.Classify(Fibo2Variants.Catalog, FileNameProbe("fibo2_edit_turbo_bf16"), []);

        Assert.Equal("dev", dev.Variant.Id);
        Assert.Equal("edit-turbo", editTurbo.Variant.Id);
        Assert.Equal(Fibo2Recipe.FamilyDefaults, recipe.DefaultsFor(dev));
        Assert.Equal(Fibo2Recipe.EditTurboDefaults, recipe.DefaultsFor(editTurbo));
    }

    [Fact]
    public void Variants_UnknownFileNameFallsBackToDev()
    {
        ResolvedModelVariant unknown = ModelVariantResolver.Classify(Fibo2Variants.Catalog, FileNameProbe("model"), []);

        Assert.Equal("dev", unknown.Variant.Id);
    }

    [Fact]
    public void Variants_CallerHintSelectsEditTurbo()
    {
        ResolvedModelVariant hinted = ModelVariantResolver.Classify(Fibo2Variants.Catalog, FileNameProbe("model"), ["edit"]);

        Assert.Equal("edit-turbo", hinted.Variant.Id);
    }

    /// <summary>A probe that carries only a file name, which is all the filename rule reads.</summary>
    private static CheckpointProbe FileNameProbe(string name) =>
        new(new HashSet<string>(StringComparer.Ordinal), new Dictionary<string, TensorShape>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), [name]);
}
