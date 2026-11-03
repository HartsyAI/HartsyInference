using HartsyInference.Engine;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The OneReward catalog entries: Comfy-Org's bf16 and fp8 builds of the FLUX.1 Fill fine-tune, with Fill's side models
/// and the object-removal LoRA. Hashes are the LFS object ids at Comfy-Org/OneReward @77a4042ca2.</summary>
public sealed class OneRewardCatalogTests
{
    [Theory]
    [InlineData("onereward-fill", "flux.1-fill-dev-OneReward-transformer_bf16.safetensors",
        "1403db15d1d182776315922334b4412ca343a26c3c0d1c77793ceda4792b20e9")]
    [InlineData("onereward-fill-fp8", "flux.1-fill-dev-OneReward-transformer_fp8.safetensors",
        "dd5cc73df9d2fb1e5e9d8ec518bcb7ea1103f55f679ad58877aa192e00481d1a")]
    public void Entry_PinsTheComfyTransformerWithFillsSideModelsAndTheLora(string id, string file, string sha)
    {
        CatalogEntry? entry = ModelCatalog.Find(id);

        Assert.NotNull(entry);
        Assert.Equal(ModelStatus.ValidationPending, entry!.Status);
        Assert.False(entry.CliDrivable);
        ModelAsset transformer = Assert.Single(entry.Assets, a => a.Role == "transformer");
        Assert.Equal("Comfy-Org/OneReward", transformer.Repo);
        Assert.Equal($"split_files/diffusion_models/{file}", transformer.RepoPath);
        Assert.Equal(sha, transformer.Sha256);
        Assert.Equal(ModelDownloader.TargetPath(transformer), ModelDownloader.PrimaryLocalPath(entry));
        Assert.Contains(SideModels.ClipL, entry.Assets);
        Assert.Contains(SideModels.T5XxlEnconly, entry.Assets);
        Assert.Contains(SideModels.FluxAe, entry.Assets);
        Assert.Contains(SideModels.OneRewardObjectRemovalLora, entry.Assets);
    }

    [Fact]
    public void ObjectRemovalLora_LandsWhereTheLoraResolverLooks()
    {
        ModelAsset lora = SideModels.OneRewardObjectRemovalLora;

        Assert.Equal("Comfy-Org/OneReward", lora.Repo);
        Assert.Equal("split_files/loras/flux.1-fill-dev-object-removal-lora.safetensors", lora.RepoPath);
        Assert.Equal("63b3f284682ffe65eff22473179b48db107b0d1ff8f2b0a03c6dd2c8f21e8451", lora.Sha256);
        Assert.Equal("lora", lora.Role);
        // LoraResolver searches Lora/ (and its spelling variants) by file name.
        Assert.Equal("Lora", lora.TargetSubdir);
    }

    [Fact]
    public void FillCatalogIds_AreCatalogEntriesThatResolveToTheFlux1Recipe()
    {
        Assert.All(Flux1Recipe.FillCatalogIds, id =>
        {
            Assert.NotNull(ModelCatalog.Find(id));
            Assert.IsType<Flux1Recipe>(RecipeRegistry.Resolve(id));
        });
    }
}
