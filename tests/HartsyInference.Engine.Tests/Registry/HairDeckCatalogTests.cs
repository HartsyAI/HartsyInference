using System.Text.RegularExpressions;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class HairDeckCatalogTests
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);

    [Fact]
    public void HairDeck_IsACliDrivableMeshEntryWhoseFirstAssetIsTheModelDirectoryMarker()
    {
        CatalogEntry? entry = ModelCatalog.Find("hairdeck");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Equal("hyz317/HairDeck", entry.HuggingFaceRepo);
        Assert.Equal("config.json", entry.Assets[0].RepoPath);
        Assert.Equal("3D/HairDeck", entry.Assets[0].TargetSubdir);
    }

    [Fact]
    public void HairDeck_ListsTheFourCheckpointsAndTheVisionModelsWithPinnedHashes()
    {
        CatalogEntry entry = ModelCatalog.Find("hairdeck")!;
        string[] checkpoints =
        [
            "ckpt/image_model/model.safetensors",
            "ckpt/color_model/model.safetensors",
            "ckpt/chain_model/model.safetensors",
            "ckpt/card_vae/model.safetensors",
        ];
        foreach (string path in checkpoints)
        {
            Assert.Contains(entry.Assets, asset => asset.RepoPath == path && asset.Repo == "hyz317/HairDeck");
        }

        Assert.Contains(entry.Assets, asset => asset.Repo == "facebook/sam3");
        Assert.Contains(entry.Assets, asset => asset.Repo == "ZhengPeng7/BiRefNet-matting");
        Assert.Contains(entry.Assets, asset => asset.Repo == "facebook/dinov3-vitl16-pretrain-lvd1689m");
        foreach (ModelAsset asset in entry.Assets)
        {
            Assert.NotNull(asset.Sha256);
            Assert.Matches(Sha256, asset.Sha256!);
        }
    }

    [Fact]
    public void HairDeck_AssetsHaveDistinctDestinations()
    {
        CatalogEntry entry = ModelCatalog.Find("hairdeck")!;

        IEnumerable<string> destinations = entry.Assets.Select(asset => Path.Combine(asset.TargetSubdir, asset.FileName));

        Assert.Equal(entry.Assets.Count, destinations.Distinct(StringComparer.OrdinalIgnoreCase).Count());
    }
}
