using HartsyInference.Engine;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Fibo 2.0 catalog entry: unverified, drivable from the CLI by id, and pinned to the Comfy-Org files with no hashes yet.</summary>
public sealed class Fibo2CatalogTests
{
    [Fact]
    public void CatalogEntry_IsUnverifiedAndCliDrivable()
    {
        CatalogEntry? entry = ModelCatalog.Find("fibo2-dev");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Image, entry!.Modality);
        Assert.Equal(ModelStatus.ValidationPending, entry.Status);
        Assert.True(entry.CliDrivable);
    }

    [Fact]
    public void CatalogEntry_PinsTheThreeComfyFilesWithHashes()
    {
        CatalogEntry entry = ModelCatalog.Find("fibo2-dev")!;

        ModelAsset transformer = Assert.Single(entry.Assets, asset => asset.Role == "transformer");
        Assert.Equal("Comfy-Org/Fibo-2.0", transformer.Repo);
        Assert.Equal("diffusion_models/fibo2_dev_bf16.safetensors", transformer.RepoPath);
        Assert.Equal("2a09866ff073cae9114f5fd93c0f530f84d41f74dc06e3d0c1372587bb3e9984", transformer.Sha256);

        ModelAsset encoder = Assert.Single(entry.Assets, asset => asset.Role == "text encoder");
        Assert.Equal("text_encoders/qwen3vl_4b_bf16.safetensors", encoder.RepoPath);
        Assert.Equal("fibo2/qwen3vl_4b_bf16.safetensors", encoder.TargetName);
        Assert.Equal("36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34", encoder.Sha256);

        ModelAsset vae = Assert.Single(entry.Assets, asset => asset.Role == "vae");
        Assert.Equal("vae/flux2-vae.safetensors", vae.RepoPath);
        Assert.Equal("868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3", vae.Sha256);
    }

    [Fact]
    public void CatalogEntry_ResolvesToTheFiboRecipe()
    {
        Assert.IsType<Fibo2Recipe>(RecipeRegistry.Resolve("fibo2-dev"));
    }
}
