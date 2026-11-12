using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>How the FLUX.2 recipe tells the Iris-Z builds from their FLUX.2 bases and what it declares for them.</summary>
public sealed class IrisZVariantTests
{
    [Theory]
    [InlineData(4096, true, "iris-z-small")]
    [InlineData(6144, true, "iris-z-big")]
    [InlineData(4096, false, "klein-9b")]
    [InlineData(6144, false, "dev")]
    [InlineData(3072, false, "klein-4b")]
    public void TheModalityEmbeddingSeparatesIrisZFromItsBase(int hidden, bool modality, string expected)
    {
        ResolvedModelVariant variant = ModelVariantResolver.Classify(Flux2Variants.Catalog, Probe(hidden, modality), []);

        Assert.Equal(expected, variant.Id);
        Assert.Equal(ModelVariantSource.Structure, variant.Source);
    }

    [Fact]
    public void AHintCannotSelectIrisZWithoutItsTensor()
    {
        ResolvedModelVariant variant = ModelVariantResolver.Classify(Flux2Variants.Catalog, Probe(4096, modality: false), ["iris-z-small"]);

        Assert.Equal("klein-9b", variant.Id);
    }

    [Fact]
    public void ConfigsFollowTheBasesWithFourModalities()
    {
        Flux2Config small = Flux2Variants.ConfigFor(Resolved(Flux2Variants.IrisZSmall));
        Flux2Config big = Flux2Variants.ConfigFor(Resolved(Flux2Variants.IrisZBig));

        Assert.Equal((4096, 8, 24, 12288, 4), (small.HiddenSize, small.Depth, small.DepthSingleBlocks, small.ContextInDim, small.ModalityCount));
        Assert.Equal((6144, 8, 48, 15360, 4), (big.HiddenSize, big.Depth, big.DepthSingleBlocks, big.ContextInDim, big.ModalityCount));
        Assert.False(small.GuidanceEmbed);
        Assert.False(big.GuidanceEmbed);
        Assert.Equal(Flux2TextEncoderType.Mistral, big.TextEncoderType);
    }

    [Theory]
    [InlineData("iris-z-small")]
    [InlineData("iris-z-big")]
    [InlineData("flux2")]
    public void TheFluxTwoRecipeServesTheCatalogIds(string familyId)
    {
        Assert.True(new Flux2Recipe().Matches(familyId));
        Assert.Same(RecipeRegistry.Resolve("flux2"), RecipeRegistry.Resolve(familyId));
    }

    [Fact]
    public void IrisZDeclaresItsOwnDefaultsAndFeatures()
    {
        Flux2Recipe recipe = new();
        ResolvedModelVariant small = Resolved(Flux2Variants.IrisZSmall);
        ResolvedModelVariant big = Resolved(Flux2Variants.IrisZBig);

        Assert.Equal((50, 4.0f), (recipe.DefaultsFor(small).Steps, recipe.DefaultsFor(small).CfgScale));
        Assert.Equal((50, 3.5f), (recipe.DefaultsFor(big).Steps, recipe.DefaultsFor(big).CfgScale));
        Assert.Equal(ImageFeatures.RefEdit | ImageFeatures.Lora, recipe.SupportsFor(small));
        Assert.Equal(recipe.Supports, recipe.SupportsFor(Resolved(Flux2Variants.Klein9B)));
        Assert.Equal(8, recipe.DefaultsFor(Resolved(Flux2Variants.Klein9B)).Steps);
    }

    private static ResolvedModelVariant Resolved(ModelVariant variant) =>
        new("flux2", variant, ModelVariantSource.Structure, "test");

    private static CheckpointProbe Probe(int hidden, bool modality)
    {
        Dictionary<string, TensorShape> shapes = new(StringComparer.Ordinal)
        {
            ["img_in.weight"] = new TensorShape(hidden, 128),
        };
        if (modality)
        {
            shapes["modality_in.weight"] = new TensorShape(4, hidden);
        }
        return new CheckpointProbe(new HashSet<string>(shapes.Keys, StringComparer.Ordinal), shapes,
            new Dictionary<string, string>(), []);
    }
}
