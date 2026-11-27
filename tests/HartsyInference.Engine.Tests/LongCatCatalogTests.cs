using HartsyInference.Engine;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The LongCat-Video-Avatar catalog row: the six 1.5 bf16 shards carry their HF LFS hashes in the role the loader expects, the index and
/// config are listed without a hash (git blobs), the shared text encoder and VAE are the engine's side models, and the Whisper front end is not a
/// row asset (the recipe reads the audio stack's cached copy at the pinned revision).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatCatalogTests
{
    [Fact]
    public void TheAvatarRowDeclaresEachFileWithItsRole()
    {
        CatalogEntry entry = ModelCatalog.Find("longcat-avatar")!;

        Assert.Equal(Modality.Video, entry.Modality);
        Assert.Equal("meituan-longcat/LongCat-Video-Avatar-1.5", entry.HuggingFaceRepo);
        List<ModelAsset> transformer = entry.Assets.Where(a => a.Role == "transformer").ToList();
        Assert.Equal(6, transformer.Count);
        Assert.All(transformer, shard =>
        {
            Assert.StartsWith("base_model/diffusion_pytorch_model-", shard.RepoPath, StringComparison.Ordinal);
            Assert.Equal(64, shard.Sha256!.Length);
            Assert.Equal("LongCat/avatar-1.5/base_model", shard.TargetSubdir);
        });
        List<ModelAsset> config = entry.Assets.Where(a => a.Role == "config").ToList();
        Assert.Equal(["base_model/config.json", "base_model/diffusion_pytorch_model.safetensors.index.json"],
            config.Select(a => a.RepoPath).Order(StringComparer.Ordinal).ToArray());
        Assert.All(config, file => Assert.Null(file.Sha256));
        Assert.Contains(entry.Assets, a => a.Repo == SideModels.Umt5Xxl.Repo && a.RepoPath == SideModels.Umt5Xxl.RepoPath);
        Assert.Contains(entry.Assets, a => a.Repo == SideModels.Wan21Vae.Repo && a.RepoPath == SideModels.Wan21Vae.RepoPath);
        Assert.DoesNotContain(entry.Assets, a => a.Repo == "openai/whisper-large-v3");
    }

    [Fact]
    public void TheAvatarRowPutsTheDmdLoraWhereTheRecipeReadsIt()
    {
        CatalogEntry entry = ModelCatalog.Find("longcat-avatar")!;
        ModelAsset lora = Assert.Single(entry.Assets, a => a.Role == "lora");
        ModelAsset primary = ModelDownloader.PrimaryAsset(entry)!;
        string root = Path.GetFullPath("models");
        // The checkpoint's folder, which is what the recipe resolves the primary shard to.
        string checkpoint = Path.Combine(root, primary.TargetSubdir);

        string downloaded = Path.Combine(root, lora.TargetSubdir, lora.TargetName!);

        Assert.Equal("lora/dmd_lora.safetensors", lora.RepoPath);
        Assert.Equal(64, lora.Sha256!.Length);
        Assert.Equal(Path.GetFullPath(LongCatRecipeSupport.DmdLoraPath(checkpoint)), Path.GetFullPath(downloaded));
    }

    [Fact]
    public void TheFirstShardIsTheRowsPrimaryAssetSoTheCheckpointFolderResolves()
    {
        CatalogEntry entry = ModelCatalog.Find("longcat-avatar")!;

        ModelAsset primary = ModelDownloader.PrimaryAsset(entry)!;

        Assert.Equal("base_model/diffusion_pytorch_model-00001-of-00006.safetensors", primary.RepoPath);
    }
}
