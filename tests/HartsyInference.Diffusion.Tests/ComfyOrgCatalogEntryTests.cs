using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Comfy-Org catalog entries for families the engine already runs: Flux.1 Fill, Flux.2 Klein 4B and the Wan 2.2
/// A14B pairs. The hashes are the Hugging Face LFS object ids at Comfy-Org/Flux-1-Dev @10cf201a3d, Flux-2-Klein-4B @5b3abca5c0
/// and Wan-2.2 @ee6f4a4073.</summary>
public sealed class ComfyOrgCatalogEntryTests
{
    [Theory]
    [InlineData("flux1-fill")]
    [InlineData("flux2-klein-4b")]
    [InlineData("wan-22-t2v-a14b")]
    [InlineData("wan-22-i2v-a14b")]
    public void NewEntries_AreUnverifiedAndNotClaimedCliDrivable(string id)
    {
        CatalogEntry? entry = ModelCatalog.Find(id);

        Assert.NotNull(entry);
        Assert.Equal(ModelStatus.ValidationPending, entry!.Status);
        Assert.False(entry.CliDrivable);
    }

    [Fact]
    public void FluxFill_PinsTheComfyTransformerAndTheSideModelsItOmits()
    {
        CatalogEntry entry = ModelCatalog.Find("flux1-fill")!;

        ModelAsset transformer = Assert.Single(entry.Assets, a => a.Role == "transformer");
        Assert.Equal("Comfy-Org/Flux-1-Dev", transformer.Repo);
        Assert.Equal("split_files/diffusion_models/flux1-fill-dev.safetensors", transformer.RepoPath);
        Assert.Equal("Stable-Diffusion/Flux", transformer.TargetSubdir);
        Assert.Equal("03e289f530df51d014f48e675a9ffa2141bc003259bf5f25d75b957e920a41ca", transformer.Sha256);
        Assert.Contains(SideModels.ClipL, entry.Assets);
        Assert.Contains(SideModels.T5XxlEnconly, entry.Assets);
        Assert.Contains(SideModels.FluxAe, entry.Assets);
    }

    [Fact]
    public void FluxFill_ResolvesToTheFlux1Recipe()
    {
        Assert.IsType<Flux1Recipe>(RecipeRegistry.Resolve("flux1-fill"));
    }

    [Fact]
    public void Klein4B_PinsTheComfyTransformerWithQwen3_4BAndTheFlux2Vae()
    {
        CatalogEntry entry = ModelCatalog.Find("flux2-klein-4b")!;

        ModelAsset transformer = Assert.Single(entry.Assets, a => a.Role == "transformer");
        Assert.Equal("Comfy-Org/Flux-2-Klein-4B", transformer.Repo);
        Assert.Equal("split_files/diffusion_models/flux-2-klein-4b.safetensors", transformer.RepoPath);
        Assert.Equal("Stable-Diffusion/Flux2", transformer.TargetSubdir);
        Assert.Equal("ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343", transformer.Sha256);
        Assert.Contains(SideModels.Qwen3_4B, entry.Assets);
        Assert.Contains(SideModels.Flux2Vae, entry.Assets);
    }

    [Fact]
    public void Klein4B_ResolvesToTheFlux2Recipe()
    {
        Assert.IsType<Flux2Recipe>(RecipeRegistry.Resolve("flux2-klein-4b"));
    }

    [Theory]
    [InlineData("wan-22-t2v-a14b", "t2v", "cad711ae211c8b23455ec68cd6a190a33a3d874234a77eb57266d73f8f0e6c9f",
        "e71b96d7c82e638694c5e7fb98fac4bfb0e4ddc5fbbb4b1df40da8f0f1278a97")]
    [InlineData("wan-22-i2v-a14b", "i2v", "6122e79d55e0f235698d11d657f3b196c5273c830da00b2b013c5a048d5e6a42",
        "5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e")]
    public void WanA14B_PinsTheHighNoisePrimaryAndTheLowNoiseExpert(string id, string task, string highSha, string lowSha)
    {
        CatalogEntry entry = ModelCatalog.Find(id)!;

        Assert.Equal(Modality.Video, entry.Modality);
        ModelAsset high = Assert.Single(entry.Assets, a => a.Role == "transformer");
        Assert.Equal("Comfy-Org/Wan-2.2", high.Repo);
        Assert.Equal($"split_files/diffusion_models/wan2.2_{task}_high_noise_14B_fp8_scaled.safetensors", high.RepoPath);
        Assert.Equal(highSha, high.Sha256);
        ModelAsset low = Assert.Single(entry.Assets, a => a.Role == ModelAsset.LowNoiseExpertRole);
        Assert.Equal("Comfy-Org/Wan-2.2", low.Repo);
        Assert.Equal($"split_files/diffusion_models/wan2.2_{task}_low_noise_14B_fp8_scaled.safetensors", low.RepoPath);
        Assert.Equal(lowSha, low.Sha256);
        Assert.All(entry.Assets, a => Assert.Equal("Stable-Diffusion/Wan", a.TargetSubdir));
        // The high-noise file is what the entry loads; the expert only rides along as the swap model.
        Assert.Equal(ModelDownloader.TargetPath(high), ModelDownloader.PrimaryLocalPath(entry));
    }

    [Theory]
    [InlineData("wan-22-t2v-a14b")]
    [InlineData("wan-22-i2v-a14b")]
    public void WanA14B_ResolvesToTheGenericWanRecipe(string id)
    {
        IVideoRecipe? recipe = VideoRecipeRegistry.Resolve(id);

        Assert.IsType<WanVideoRecipe>(recipe);
        Assert.Equal("wan", recipe!.Name);
    }

    [Fact]
    public void SwapExpert_IsPairedWhenTheEntryRunsItsOwnHighNoiseFile()
    {
        CatalogEntry entry = ModelCatalog.Find("wan-22-t2v-a14b")!;
        ModelAsset expert = Assert.Single(entry.Assets, a => a.Role == ModelAsset.LowNoiseExpertRole);

        VideoRequest? paired = InferenceEngine.WithCatalogSwapExpert(Spec(entry, ModelDownloader.PrimaryLocalPath(entry)), Request());

        Assert.Equal(ModelDownloader.TargetPath(expert), paired!.VideoSwapModel);
    }

    [Fact]
    public void SwapExpert_KeepsTheCallersOwnSwapModel()
    {
        CatalogEntry entry = ModelCatalog.Find("wan-22-t2v-a14b")!;
        VideoRequest request = Request() with { VideoSwapModel = "my_low_noise.safetensors" };

        VideoRequest? result = InferenceEngine.WithCatalogSwapExpert(Spec(entry, ModelDownloader.PrimaryLocalPath(entry)), request);

        Assert.Equal("my_low_noise.safetensors", result!.VideoSwapModel);
    }

    [Fact]
    public void SwapExpert_IsNotPairedWithAModelPathOverride()
    {
        CatalogEntry entry = ModelCatalog.Find("wan-22-t2v-a14b")!;
        string other = Path.Combine(Path.GetTempPath(), "some_other_wan_14b.safetensors");

        VideoRequest? result = InferenceEngine.WithCatalogSwapExpert(Spec(entry, other), Request());

        Assert.Null(result!.VideoSwapModel);
    }

    [Fact]
    public void SwapExpert_IsNotPairedForAnEntryWithoutOne()
    {
        CatalogEntry entry = ModelCatalog.Find("wan")!;

        VideoRequest? result = InferenceEngine.WithCatalogSwapExpert(Spec(entry, ModelDownloader.PrimaryLocalPath(entry)), Request());

        Assert.Null(result!.VideoSwapModel);
    }

    [Fact]
    public void SwapExpert_LeavesANullRequestNull()
    {
        CatalogEntry entry = ModelCatalog.Find("wan-22-t2v-a14b")!;

        Assert.Null(InferenceEngine.WithCatalogSwapExpert(Spec(entry, ModelDownloader.PrimaryLocalPath(entry)), null));
    }

    private static ModelSpec Spec(CatalogEntry entry, string? localPath) =>
        new ModelSpec { Requested = entry.Id, Modality = entry.Modality, Catalog = entry, LocalPath = localPath };

    private static VideoRequest Request() => new VideoRequest { Prompt = "a fox running through snow" };
}
