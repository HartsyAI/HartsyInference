using System.Text.RegularExpressions;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class HunyuanXPartCatalogTests
{
    [Fact]
    public void XPart_CarriesTheFourCheckpointsWithPinnedHashesUnderTheSharedPartFolder()
    {
        CatalogEntry entry = ModelCatalog.Find("hunyuan3d-xpart")!;

        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Equal<string[]>(XPartFiles(), [.. entry.Assets.Select(a => a.FileName).Order(StringComparer.Ordinal)]);
        Assert.All(entry.Assets, a => Assert.Equal("tencent/Hunyuan3D-Part", a.Repo));
        Assert.All(entry.Assets, a => Assert.Equal("3D/hunyuan3d-part", a.TargetSubdir));
        Assert.All(entry.Assets, a => Assert.Matches(new Regex("^[0-9a-f]{64}$"), a.Sha256 ?? string.Empty));
        // The P3-SAM file is the one the hunyuan3d-part entry downloads: one copy on disk serves both.
        Assert.Equal(ModelCatalog.Find("hunyuan3d-part")!.Assets[0].Sha256, entry.Assets.Single(a => a.FileName == "p3sam/p3sam.safetensors").Sha256);
    }

    [Fact]
    public void XPart_IsWiredToTheMeshServiceAsAMeshOnlyModel()
    {
        Assert.Equal(["hunyuan3d-xpart"], MeshModelIds.XPartMesh);
        Assert.Contains("hunyuan3d-xpart", MeshModelIds.All);
        Assert.DoesNotContain("hunyuan3d-xpart", MeshModelIds.ImageToMesh);
        Assert.DoesNotContain("hunyuan3d-xpart", MeshModelIds.PartMesh);
        Assert.Equal("hunyuan3d-xpart", ModelResolver.Resolve("hunyuan3d-xpart", null, Modality.Mesh).Catalog?.Id);
    }

    private static string[] XPartFiles() =>
        ["conditioner/conditioner.safetensors", "model/model.safetensors", "p3sam/p3sam.safetensors", "shapevae/shapevae.safetensors"];
}
