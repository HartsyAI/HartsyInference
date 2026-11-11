using System.Text.RegularExpressions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class Hunyuan3DCatalogTests
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);

    public static TheoryData<string> Ids => new() { "hunyuan3d", "hunyuan3d-fast", "hunyuan3d-turbo", "hunyuan3d-mini", "hunyuan3d-mv" };

    [Theory]
    [MemberData(nameof(Ids))]
    public void Build_IsAMeshEntryWhoseWeightsComeFirstAndWhoseFilesAreHashed(string id)
    {
        CatalogEntry? entry = ModelCatalog.Find(id);

        Assert.NotNull(entry);
        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.EndsWith("/model.fp16.safetensors", entry.Assets[0].RepoPath);
        Assert.Equal("3D/" + id, entry.Assets[0].TargetSubdir);
        Assert.Contains(entry.Assets, a => a.RepoPath.EndsWith("/config.yaml", StringComparison.Ordinal) && a.TargetSubdir == "3D/" + id);
        Assert.All(entry.Assets, asset => Assert.Matches(Sha256, asset.Sha256 ?? string.Empty));
    }

    [Fact]
    public void Turbo_DecodesWithFlashVdmAndSoCarriesItsTurboVae()
    {
        CatalogEntry entry = ModelCatalog.Find("hunyuan3d-turbo")!;

        ModelAsset vae = Assert.Single(entry.Assets, a => a.Role == ModelCatalog.Hunyuan3DFlashVdmVaeRole);
        Assert.Equal("hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors", vae.RepoPath);
        Assert.Equal("3D/hunyuan3d-vae-turbo", vae.TargetSubdir);
        Assert.DoesNotContain(ModelCatalog.Find("hunyuan3d-fast")!.Assets, a => a.Role == ModelCatalog.Hunyuan3DFlashVdmVaeRole);
    }

    [Fact]
    public void SingleImageBuilds_AreWiredToTheMeshServiceAndMultiviewIsNot()
    {
        Assert.All(MeshModelIds.Hunyuan3D, id => Assert.Contains(id, MeshModelIds.ImageToMesh));
        Assert.DoesNotContain("hunyuan3d-mv", MeshModelIds.All);
        Assert.False(ModelCatalog.Find("hunyuan3d-mv")!.CliDrivable);
    }

    [Theory]
    [InlineData("hunyuan3d-fast")]
    [InlineData("hunyuan3d-turbo")]
    [InlineData("hunyuan3d-mini")]
    public void Resolve_PicksTheBuildByItsCatalogId(string id)
    {
        ModelSpec spec = ModelResolver.Resolve(id, null, Modality.Mesh);

        Assert.Equal(id, spec.Catalog?.Id);
    }
}
