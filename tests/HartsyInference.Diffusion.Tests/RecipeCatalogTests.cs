using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Engine;
using HartsyInference.Engine.Catalog;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Variants;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The recipe catalog answers from the same recipes generation runs, for every family and every declared variant.</summary>
public sealed class RecipeCatalogTests
{
    private readonly RecipeCatalog _catalog = new();

    [Fact]
    public void All_CoversExactlyTheRegisteredImageAndVideoNames()
    {
        IReadOnlyList<ModelDescriptor> all = _catalog.All();

        Assert.Equal(RecipeRegistry.DefaultNames, all.OfType<ImageModelDescriptor>().Select(d => d.Recipe.FamilyId).ToList());
        Assert.Equal(VideoRecipeRegistry.DefaultNames, all.OfType<VideoModelDescriptor>().Select(d => d.Recipe.FamilyId).ToList());
        Assert.Equal(RecipeRegistry.DefaultNames.Count + VideoRecipeRegistry.DefaultNames.Count, all.Count);
    }

    [Fact]
    public void All_FiltersByModality()
    {
        Assert.All(_catalog.All(Modality.Video), descriptor => Assert.IsType<VideoModelDescriptor>(descriptor));
        Assert.Equal(VideoRecipeRegistry.DefaultNames.Count, _catalog.All(Modality.Video).Count);
        Assert.Equal(RecipeRegistry.DefaultNames.Count, _catalog.All(Modality.Image).Count);
    }

    [Fact]
    public void EveryImageFamilyDefaultsMatchItsRecipeForEveryDeclaredVariant()
    {
        foreach (string name in RecipeRegistry.DefaultNames)
        {
            IArchitectureRecipe recipe = RecipeRegistry.Resolve(name)!;
            Assert.Equal(recipe.DefaultsFor(CatalogDefault(recipe.Variants)), ((ImageModelDescriptor)_catalog.DescribeFamily(Modality.Image, name)).Defaults);
            IReadOnlyList<ModelVariant> variants = recipe.Variants?.Variants ?? [];
            foreach (ModelVariant variant in variants)
            {
                ResolvedModelVariant resolved = new(recipe.Variants!.FamilyId, variant, ModelVariantSource.CallerHint, "test");
                ImageModelDescriptor descriptor = (ImageModelDescriptor)_catalog.DescribeFamily(Modality.Image, name, variant.Id);
                Assert.Equal(recipe.DefaultsFor(resolved), descriptor.Defaults);
            }
        }
    }

    [Fact]
    public void EveryVideoFamilyDefaultsMatchItsRecipeForEveryDeclaredVariant()
    {
        foreach (string name in VideoRecipeRegistry.DefaultNames)
        {
            IVideoRecipe recipe = VideoRecipeRegistry.Resolve(name)!;
            Assert.Equal(recipe.DefaultsFor(CatalogDefault(recipe.Variants)), ((VideoModelDescriptor)_catalog.DescribeFamily(Modality.Video, name)).Defaults);
            IReadOnlyList<ModelVariant> variants = recipe.Variants?.Variants ?? [];
            foreach (ModelVariant variant in variants)
            {
                ResolvedModelVariant resolved = new(recipe.Variants!.FamilyId, variant, ModelVariantSource.CallerHint, "test");
                VideoModelDescriptor descriptor = (VideoModelDescriptor)_catalog.DescribeFamily(Modality.Video, name, variant.Id);
                Assert.Equal(recipe.DefaultsFor(resolved), descriptor.Defaults);
            }
        }
    }

    [Fact]
    public void AnUnknownFamilyThrows()
    {
        Assert.Throws<RecipeNotRegisteredException>(() => _catalog.DescribeFamily(Modality.Image, "no-such-family"));
        Assert.Throws<RecipeNotRegisteredException>(() => _catalog.DescribeFamily(Modality.Video, "no-such-family"));
    }

    [Fact]
    public void AnUnknownVariantThrowsNamingTheValidIds()
    {
        string family = RecipeRegistry.DefaultNames.First(name => RecipeRegistry.Resolve(name)!.Variants is not null);
        IReadOnlyList<string> valid = _catalog.DescribeFamily(Modality.Image, family).Variants;

        ArgumentException error = Assert.Throws<ArgumentException>(() => _catalog.DescribeFamily(Modality.Image, family, "no-such-variant"));

        Assert.NotEmpty(valid);
        foreach (string id in valid)
        {
            Assert.Contains(id, error.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void DescriptorsRoundTripThroughJsonKeepingTheDiscriminatorAndValues()
    {
        // The API's serializer options: strings for enums, and the polymorphic base type on the wire.
        JsonSerializerOptions options = new() { Converters = { new JsonStringEnumConverter() } };
        ModelDescriptor image = _catalog.DescribeFamily(Modality.Image, "sdxl");
        ModelDescriptor video = _catalog.DescribeFamily(Modality.Video, "wan");

        string imageJson = JsonSerializer.Serialize<ModelDescriptor>(image, options);
        ImageModelDescriptor imageBack = Assert.IsType<ImageModelDescriptor>(
            JsonSerializer.Deserialize<ModelDescriptor>(imageJson, options));
        Assert.Equal("image", JsonDocument.Parse(imageJson).RootElement.GetProperty("$type").GetString());
        Assert.Equal(image.Recipe, imageBack.Recipe);
        Assert.Equal(((ImageModelDescriptor)image).Defaults, imageBack.Defaults);
        Assert.Equal(((ImageModelDescriptor)image).Features, imageBack.Features);
        Assert.Equal(((ImageModelDescriptor)image).InputLimits, imageBack.InputLimits);

        string videoJson = JsonSerializer.Serialize<ModelDescriptor>(video, options);
        VideoModelDescriptor videoBack = Assert.IsType<VideoModelDescriptor>(
            JsonSerializer.Deserialize<ModelDescriptor>(videoJson, options));
        VideoModelDescriptor videoOriginal = (VideoModelDescriptor)video;
        Assert.Equal("video", JsonDocument.Parse(videoJson).RootElement.GetProperty("$type").GetString());
        Assert.Equal(videoOriginal.Recipe, videoBack.Recipe);
        Assert.Equal(videoOriginal.Defaults, videoBack.Defaults);
        Assert.Equal(videoOriginal.Constraints.Frames, videoBack.Constraints.Frames);
        Assert.Equal(videoOriginal.Features, videoBack.Features);
        Assert.Equal(videoOriginal.Sampling.Samplers, videoBack.Sampling.Samplers);
    }

    /// <summary>What a family with no variant evidence runs: its catalog's default variant, or no variant at all.</summary>
    private static ResolvedModelVariant? CatalogDefault(ModelVariantCatalog? catalog) =>
        catalog is null ? null : new ResolvedModelVariant(catalog.FamilyId, catalog.Default, ModelVariantSource.Default, "test");
}
