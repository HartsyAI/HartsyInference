using System.Text.RegularExpressions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class HunyuanPartCatalogTests
{
    [Fact]
    public void Part_IsAMeshEntryWithTheOneHashedCheckpoint()
    {
        CatalogEntry? entry = ModelCatalog.Find("hunyuan3d-part");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.True(entry.CliDrivable);
        ModelAsset asset = Assert.Single(entry.Assets);
        Assert.Equal("tencent/Hunyuan3D-Part", asset.Repo);
        Assert.Equal("p3sam/p3sam.safetensors", asset.RepoPath);
        Assert.Equal("p3sam/p3sam.safetensors", asset.FileName);
        Assert.Equal("3D/hunyuan3d-part", asset.TargetSubdir);
        Assert.Matches(new Regex("^[0-9a-f]{64}$"), asset.Sha256 ?? string.Empty);
    }

    [Fact]
    public void Part_IsWiredToTheMeshServiceAsAMeshOnlyModel()
    {
        Assert.Equal(["hunyuan3d-part"], MeshModelIds.PartMesh);
        Assert.Contains("hunyuan3d-part", MeshModelIds.All);
        Assert.DoesNotContain("hunyuan3d-part", MeshModelIds.ImageToMesh);
        Assert.DoesNotContain("hunyuan3d-part", MeshModelIds.Hunyuan3D);
        Assert.Equal("hunyuan3d-part", ModelResolver.Resolve("hunyuan3d-part", null, Modality.Mesh).Catalog?.Id);
    }
}
