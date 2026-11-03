using HartsyInference.Engine;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The MOSS-Image catalog entries: Comfy-Org's single-file Base and SFT builds (bf16 and int8 ConvRot), pinned to
/// the LFS object ids at Comfy-Org/MOSS-Image @530a4c94f9, with the side models <see cref="MossImageRecipe"/> actually falls
/// back to. The entries used to list the Q8 GGUF encoder and the full-decoder VAE, which the recipe never reads.</summary>
public sealed class MossImageCatalogTests
{
    public static TheoryData<string, string, string, string> Builds => new()
    {
        { "moss-image", "moss_image_base_bf16.safetensors", "cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f", "base" },
        { "moss-image-sft", "moss_image_sft_bf16.safetensors", "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f", "sft" },
        { "moss-image-int8", "moss_image_base_int8_convrot.safetensors", "f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f", "base" },
        { "moss-image-sft-int8", "moss_image_sft_int8_convrot.safetensors", "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f", "sft" },
    };

    [Theory]
    [MemberData(nameof(Builds))]
    public void Entry_PinsTheComfyBuild(string id, string file, string sha, string variant)
    {
        _ = variant;
        CatalogEntry? entry = ModelCatalog.Find(id);

        Assert.NotNull(entry);
        Assert.Equal(Modality.Image, entry!.Modality);
        Assert.Equal(ModelStatus.ValidationPending, entry.Status);
        Assert.True(entry.CliDrivable);
        ModelAsset transformer = Assert.Single(entry.Assets, a => a.Role == "transformer");
        Assert.Equal("Comfy-Org/MOSS-Image", transformer.Repo);
        Assert.Equal($"diffusion_models/{file}", transformer.RepoPath);
        Assert.Equal(sha, transformer.Sha256);
        Assert.Equal(ModelDownloader.TargetPath(transformer), ModelDownloader.PrimaryLocalPath(entry));
    }

    [Theory]
    [MemberData(nameof(Builds))]
    public void Entry_CarriesTheSideModelsTheRecipeFallsBackTo(string id, string file, string sha, string variant)
    {
        _ = (file, sha, variant);
        CatalogEntry entry = ModelCatalog.Find(id)!;

        Assert.Contains(SideModels.Qwen35_9BBf16, entry.Assets);
        Assert.Contains(SideModels.Qwen35Tokenizer, entry.Assets);
        Assert.Contains(SideModels.Flux2VaeSmall, entry.Assets);
        Assert.DoesNotContain(SideModels.Qwen35_9B, entry.Assets);
        Assert.DoesNotContain(SideModels.Flux2Vae, entry.Assets);
        Assert.All(entry.Assets, a => Assert.False(string.IsNullOrWhiteSpace(a.Sha256), $"{a.RepoPath} has no hash"));
    }

    [Theory]
    [MemberData(nameof(Builds))]
    public void Entry_ResolvesToTheRecipeAndTheVariantItsFileNames(string id, string file, string sha, string variant)
    {
        _ = sha;
        CatalogEntry entry = ModelCatalog.Find(id)!;

        Assert.IsType<MossImageRecipe>(RecipeRegistry.Resolve(id));
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(MossImageVariants.Catalog,
            CheckpointProbe.Empty with { FileNames = [Path.GetFileNameWithoutExtension(file)] }, [entry.Id]);
        Assert.Equal(variant, resolved.Variant.Id);
    }

    [Fact]
    public void RecipeIdList_MatchesTheCatalogBuilds()
    {
        // The recipe's id list is kept by hand; a Comfy build missing from it would resolve to no recipe.
        string[] builds = ModelCatalog.All
            .Where(e => e.Assets.Any(a => a.Role == "transformer" && a.Repo == "Comfy-Org/MOSS-Image"))
            .Select(e => e.Id)
            .Where(id => id != "moss-image")
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(MossImageRecipe.BuildCatalogIds.Order(StringComparer.Ordinal), builds);
    }
}
