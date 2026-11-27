using HartsyInference.Engine;
using HartsyInference.Engine.Catalog;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Variants;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The LongCat-Video-Avatar catalog row: the six 1.5 bf16 shards carry their HF LFS hashes in the role the loader expects, the index and
/// config are listed without a hash (git blobs), the shared text encoder and VAE are the engine's side models, and the Whisper front end is not a
/// row asset (the recipe reads the audio stack's cached copy at the pinned revision).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatCatalogTests
{
    [Fact]
    public void TheBaseRowDeclaresTheSixShardsTheirIndexAndConfigAndTheSideModels()
    {
        CatalogEntry entry = ModelCatalog.Find("longcat-video")!;

        Assert.Equal(Modality.Video, entry.Modality);
        Assert.Equal("meituan-longcat/LongCat-Video", entry.HuggingFaceRepo);
        List<ModelAsset> transformer = entry.Assets.Where(a => a.Role == "transformer").ToList();
        Assert.Equal(6, transformer.Count);
        Assert.All(transformer, shard =>
        {
            Assert.StartsWith("dit/diffusion_pytorch_model-", shard.RepoPath, StringComparison.Ordinal);
            Assert.Equal(64, shard.Sha256!.Length);
        });
        Assert.Equal(6, transformer.Select(shard => shard.Sha256).Distinct().Count());
        List<ModelAsset> config = entry.Assets.Where(a => a.Role == "config").ToList();
        Assert.Equal(["dit/config.json", "dit/diffusion_pytorch_model.safetensors.index.json"],
            config.Select(a => a.RepoPath).Order(StringComparer.Ordinal).ToArray());
        // The recipe reads one folder: every DiT file lands in it.
        Assert.Equal("LongCat/LongCat-Video/dit", Assert.Single(transformer.Concat(config).Select(a => a.TargetSubdir).Distinct()));
        Assert.Contains(entry.Assets, a => a.Repo == SideModels.Umt5Xxl.Repo && a.RepoPath == SideModels.Umt5Xxl.RepoPath);
        Assert.Contains(entry.Assets, a => a.Repo == SideModels.Wan21Vae.Repo && a.RepoPath == SideModels.Wan21Vae.RepoPath);
        Assert.DoesNotContain(entry.Assets, a => a.Role == "lora");
        Assert.Equal("dit/diffusion_pytorch_model-00001-of-00006.safetensors", ModelDownloader.PrimaryAsset(entry)!.RepoPath);
    }

    [Fact]
    public void TheBaseIdSelectsItsRowAndItsRecipe()
    {
        ModelSpec spec = ModelResolver.Resolve("longcat-video", null, Modality.Video);

        Assert.Equal("longcat-video", spec.Catalog!.Id);
        Assert.Equal("longcat-video", VideoRecipeRegistry.Resolve(spec.Catalog.Id)!.Name);
    }

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

    [Theory]
    [InlineData("longcat-avatar", "1.5", "LongCat/avatar-1.5/base_model", 6)]
    [InlineData("longcat-avatar-int8", "1.5", "LongCat/avatar-1.5/base_model_int8", 4)]
    [InlineData("longcat-avatar-1.6", "1.6", "LongCat/avatar-1.6/base_model_int8", 4)]
    [InlineData("longcat-avatar-1.6-bf16", "1.6", "LongCat/avatar-1.6/base_model", 6)]
    public void EachAvatarRowResolvesToTheAvatarRecipeItsBuildAndItsLoraFiles(string id, string build, string folder, int shards)
    {
        CatalogEntry entry = ModelCatalog.Find(id)!;
        ModelSpec spec = ModelResolver.Resolve(id, null, Modality.Video);
        ModelAsset primary = ModelDownloader.PrimaryAsset(entry)!;
        string root = Path.GetFullPath("models");
        // The checkpoint's folder, which is what the recipe resolves the primary shard to.
        string checkpoint = Path.Combine(root, primary.TargetSubdir);

        (IVideoRecipe? recipe, ResolvedModelVariant? variant) = RecipeResolution.ResolveVideo(spec);

        Assert.Contains(id, LongCatAvatarVariants.RowIds);
        Assert.Equal("longcat-avatar", recipe!.Name);
        Assert.Equal(build, variant!.Variant.Id);
        Assert.Equal(folder, primary.TargetSubdir);
        List<ModelAsset> transformer = entry.Assets.Where(a => a.Role == "transformer").ToList();
        Assert.Equal(shards, transformer.Count);
        Assert.Equal(shards, transformer.Select(a => a.Sha256).Distinct().Count());
        Assert.All(transformer, shard => Assert.Equal((64, folder), (shard.Sha256!.Length, shard.TargetSubdir)));
        // The recipe reads one folder: the index and configs land beside the shards.
        Assert.All(entry.Assets.Where(a => a.Role == "config"), file => Assert.Equal(folder, file.TargetSubdir));
        Assert.Contains(entry.Assets, a => a.Role == "config" && a.RepoPath.EndsWith(".safetensors.index.json", StringComparison.Ordinal));
        string[] downloaded = [.. entry.Assets.Where(a => a.Role == "lora")
            .Select(a => Path.GetFullPath(Path.Combine(root, a.TargetSubdir, a.TargetName!))).Order(StringComparer.Ordinal)];
        LongCatRecipeSupport.LongCatLoraFiles mixed = LongCatRecipeSupport.Avatar16LoraFiles(checkpoint);
        string[] read = build == "1.6" ? [mixed.DmdPath, mixed.OpsdPath] : [LongCatRecipeSupport.DmdLoraPath(checkpoint)];
        Assert.Equal(read.Select(Path.GetFullPath).Order(StringComparer.Ordinal), downloaded);
    }

    [Fact]
    public void TheFirstShardIsTheRowsPrimaryAssetSoTheCheckpointFolderResolves()
    {
        CatalogEntry entry = ModelCatalog.Find("longcat-avatar")!;

        ModelAsset primary = ModelDownloader.PrimaryAsset(entry)!;

        Assert.Equal("base_model/diffusion_pytorch_model-00001-of-00006.safetensors", primary.RepoPath);
    }
}
