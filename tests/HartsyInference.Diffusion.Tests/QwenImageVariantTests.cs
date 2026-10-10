using HartsyInference.Engine;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Qwen-Image base, Edit and Edit-2509 share every tensor, so the variant is all that stands between an edit
/// request and a text-to-image model conditioned on an edit template it never saw — a plausible, wrong image.</summary>
public sealed class QwenImageVariantTests
{
    [Theory]
    [InlineData("qwen-image-edit-plus", "edit-plus")]
    [InlineData("edit", "edit")]
    public void SwarmUiClassOrShortHint_SelectsTheVariant(string hint, string expected)
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(QwenImageVariants.Catalog, CheckpointProbe.Empty, [hint]);
        Assert.Equal(expected, resolved.Id);
    }

    /// <summary>SwarmUI's generic <c>qwen-image</c> class is auto-assigned to every unlabelled Qwen file, Edit v1
    /// included, so it must not be read as a positive "base" claim.</summary>
    [Fact]
    public void GenericFamilyClass_IsNotABaseClaim()
    {
        CheckpointProbe probe = CheckpointProbe.Empty with { FileNames = ["qwen_image_edit_fp8_e4m3fn"] };
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(QwenImageVariants.Catalog, probe, ["qwen-image", "qwen-image"]);
        Assert.True(resolved.Is(QwenImageVariants.Edit));
        Assert.Equal(ModelVariantSource.Filename, resolved.Source);
    }

    [Fact]
    public void The2511Marker_ProvesEditPlus_WhateverTheHint()
    {
        CheckpointProbe probe = CheckpointProbe.Empty with { Keys = new HashSet<string>([QwenImageVariants.TimestepZeroMarker]) };
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(QwenImageVariants.Catalog, probe, ["qwen-image-edit"]);
        Assert.True(resolved.Is(QwenImageVariants.EditPlus));
        Assert.Equal(ModelVariantSource.Structure, resolved.Source);
    }

    [Theory]
    [InlineData("qwen_image_edit_2509_fp8_e4m3fn", "edit-plus")]
    [InlineData("qwen_image_edit_fp8_e4m3fn", "edit")]
    [InlineData("qwen_image_fp8_e4m3fn", "base")]
    public void FilenameLastResort(string fileName, string expected)
    {
        CheckpointProbe probe = CheckpointProbe.Empty with { FileNames = [fileName] };
        Assert.Equal(expected, ModelVariantResolver.Classify(QwenImageVariants.Catalog, probe, []).Id);
    }

    [Fact]
    public void Base_DoesNotDeclareReferenceEditing_EditVariantsDo()
    {
        IArchitectureRecipe recipe = RecipeRegistry.Resolve("qwen-image")!;
        ImageFeatures baseFeatures = recipe.SupportsFor(Resolve([]));
        Assert.Equal(ImageFeatures.None, baseFeatures & ImageFeatures.RefEdit);
        Assert.NotEqual(ImageFeatures.None, baseFeatures & ImageFeatures.Img2Img);
        foreach (string hint in new[] { "edit", "edit-plus" })
        {
            ImageFeatures features = recipe.SupportsFor(Resolve([hint]));
            Assert.NotEqual(ImageFeatures.None, features & ImageFeatures.RefEdit);
            Assert.NotEqual(ImageFeatures.None, features & ImageFeatures.Img2Img);
        }
    }

    /// <summary>A family:variant selector reaches the variant through the same query hosts call.</summary>
    [Fact]
    public void SelectorSuffix_ReachesTheCapabilityQuery()
    {
        Engine.Dispatch.ModelSpec spec = new Engine.Dispatch.ModelSpec { Requested = "qwen-image:edit", Modality = Modality.Image };
        Assert.Equal("edit", ModelCapabilities.ResolveVariant(spec)?.Id);
        Assert.NotEqual(ImageFeatures.None, ModelCapabilities.ImageFeaturesFor(spec) & ImageFeatures.RefEdit);
        Engine.Dispatch.ModelSpec swarm = new Engine.Dispatch.ModelSpec { Requested = "qwen-image", Modality = Modality.Image, Variant = "qwen-image" };
        Assert.Equal(ImageFeatures.None, ModelCapabilities.ImageFeaturesFor(swarm) & ImageFeatures.RefEdit);
    }

    /// <summary>The repacker stamps the identity catalog's class; the resolver must read that stamp back as the same variant.</summary>
    [Fact]
    public void IdentityCatalogStamps_RoundTripThroughTheResolver()
    {
        ArtifactIdentity identity = ModelIdentityCatalog.Find("qwen-image")!;
        foreach (KeyValuePair<string, string> entry in identity.VariantClassIds)
        {
            CheckpointProbe probe = CheckpointProbe.Empty with
            {
                Metadata = new Dictionary<string, string> { [ModelVariantResolver.ArchitectureMetadataKey] = entry.Value },
            };
            Assert.Equal(entry.Key, ModelVariantResolver.Classify(QwenImageVariants.Catalog, probe, []).Id);
        }
    }

    private static ResolvedModelVariant Resolve(string[] hints) =>
        ModelVariantResolver.Classify(QwenImageVariants.Catalog, CheckpointProbe.Empty, hints);
}
