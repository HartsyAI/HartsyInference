using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The LongCat-Video-Avatar catalog row: the upstream 1.5 bf16 shards are pinned by the HF tree's LFS hashes, the index and config are listed
/// without a hash (they are git blobs, not LFS), and the side models are the same objects the recipe resolves, so each file is fetched once.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatCatalogTests
{
    [Fact]
    public void TheAvatarRowListsTheSixPinnedShardsAndTheSideModels()
    {
        CatalogEntry entry = ModelCatalog.Find("longcat-avatar")!;

        Assert.Equal(Modality.Video, entry.Modality);
        Assert.Equal("meituan-longcat/LongCat-Video-Avatar-1.5", entry.HuggingFaceRepo);
        Assert.Equal(11, entry.Assets.Count);
        List<ModelAsset> shards = entry.Assets.Where(a => a.RepoPath.EndsWith(".safetensors", StringComparison.Ordinal)
            && a.RepoPath.StartsWith("base_model/", StringComparison.Ordinal)).ToList();
        Assert.Equal(6, shards.Count);
        Assert.All(shards, shard => Assert.Equal(64, shard.Sha256!.Length));
        Assert.Contains(entry.Assets, a => a.RepoPath == "base_model/diffusion_pytorch_model.safetensors.index.json" && a.Sha256 is null);
        Assert.Contains(entry.Assets, a => a.RepoPath == "base_model/config.json" && a.Sha256 is null);
        Assert.Contains(entry.Assets, a => a.Repo == SideModels.WhisperLargeV3.Repo && a.Sha256 == SideModels.WhisperLargeV3.Sha256);
        Assert.Contains(entry.Assets, a => a.Repo == SideModels.Umt5Xxl.Repo);
        Assert.Contains(entry.Assets, a => a.Repo == SideModels.Wan21Vae.Repo);
    }

    [Fact]
    public void TheFirstShardIsTheRowsPrimaryAssetSoTheCheckpointFolderResolves()
    {
        CatalogEntry entry = ModelCatalog.Find("longcat-avatar")!;

        ModelAsset primary = ModelDownloader.PrimaryAsset(entry)!;

        Assert.Equal("base_model/diffusion_pytorch_model-00001-of-00006.safetensors", primary.RepoPath);
    }
}
