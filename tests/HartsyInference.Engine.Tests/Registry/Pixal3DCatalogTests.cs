using System.Text.RegularExpressions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class Pixal3DCatalogTests
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);

    [Fact]
    public void Entry_IsAHashedMeshEntryInTheSharedFolderLayout()
    {
        CatalogEntry? entry = ModelCatalog.Find("pixal3d");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Equal(ModelStatus.Structural, entry.Status);
        Assert.All(entry.Assets, asset => Assert.Matches(Sha256, asset.Sha256 ?? string.Empty));
        // pipeline.json, three flows with their configs, and the two decoders.
        Assert.Equal(9, entry.Assets.Count(a => a.TargetSubdir == "3D/pixal3d"));
        Assert.Contains(entry.Assets, a => a.RepoPath == "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors" && a.Repo == "TencentARC/Pixal3D");
        // NAF is a GitHub release asset the catalog cannot fetch, so no entry may pretend to carry it.
        Assert.DoesNotContain(entry.Assets, a => a.RepoPath.Contains("naf", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Entry_SharesTheDinoAndMattingFilesWithTrellis2()
    {
        CatalogEntry pixal3d = ModelCatalog.Find("pixal3d")!, trellis2 = ModelCatalog.Find("trellis2")!;

        ModelAsset dino = Assert.Single(pixal3d.Assets, a => a.Repo.StartsWith("facebook/dinov3", StringComparison.Ordinal));
        ModelAsset other = Assert.Single(trellis2.Assets, a => a.Repo.StartsWith("facebook/dinov3", StringComparison.Ordinal));
        Assert.Equal(other.Sha256, dino.Sha256);
        Assert.Equal(other.TargetSubdir, dino.TargetSubdir);
        Assert.Equal(other.RepoPath, dino.RepoPath);
        ModelAsset matter = Assert.Single(pixal3d.Assets, a => a.Repo.StartsWith("ZhengPeng7/BiRefNet", StringComparison.Ordinal));
        Assert.Equal(Assert.Single(trellis2.Assets, a => a.Repo.StartsWith("ZhengPeng7/BiRefNet", StringComparison.Ordinal)).Sha256, matter.Sha256);
    }

    [Fact]
    public void Pixal3D_IsWiredToTheMeshServiceAsAnImageModel()
    {
        Assert.Contains("pixal3d", MeshModelIds.ImageToMesh);
        Assert.Contains("pixal3d", MeshModelIds.All);
        Assert.Equal("pixal3d", ModelResolver.Resolve("pixal3d", null, Modality.Mesh).Catalog?.Id);
    }

    [Fact]
    public void MeshRequest_CarriesTheFieldOfViewAsOptional()
    {
        Assert.Null(new MeshRequest().FieldOfViewRadians);
        Assert.Equal(0.5, new MeshRequest { FieldOfViewRadians = 0.5 }.FieldOfViewRadians);
    }
}
