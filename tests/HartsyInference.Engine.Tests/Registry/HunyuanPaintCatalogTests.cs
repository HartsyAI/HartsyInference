using System.Text.RegularExpressions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class HunyuanPaintCatalogTests
{
    private static readonly Regex Sha256 = new("^[0-9a-f]{64}$", RegexOptions.Compiled);

    [Theory]
    [InlineData("hunyuan3d-paint", "tencent/Hunyuan3D-2", ".safetensors")]
    [InlineData("hunyuan3d-paint21", "tencent/Hunyuan3D-2.1", ".bin")]
    public void Paint_IsAMeshEntryWithTheUnetFirstAndEveryFileHashed(string id, string repo, string extension)
    {
        CatalogEntry? entry = ModelCatalog.Find(id);

        Assert.NotNull(entry);
        Assert.Equal(Modality.Mesh, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Equal(repo, entry.Assets[0].Repo);
        Assert.Equal("unet/diffusion_pytorch_model" + extension, entry.Assets[0].FileName);
        Assert.Contains(entry.Assets, a => a.FileName == "vae/diffusion_pytorch_model" + extension);
        Assert.All(entry.Assets, asset => Assert.Equal("3D/" + id, asset.TargetSubdir));
        Assert.All(entry.Assets, asset => Assert.Matches(Sha256, asset.Sha256 ?? string.Empty));
    }

    [Fact]
    public void PaintPbr_AlsoCarriesTheDinoEncoderButNotTheOptInUpscaler()
    {
        CatalogEntry entry = ModelCatalog.Find("hunyuan3d-paint21")!;

        Assert.Contains(entry.Assets, a => a.Repo == "facebook/dinov2-giant" && a.FileName == "dino/model.safetensors");
        Assert.DoesNotContain(entry.Assets, a => a.FileName == "RealESRGAN_x4plus.pth");
    }

    [Fact]
    public void PaintModels_AreWiredToTheMeshServiceAsMeshAndImageModels()
    {
        Assert.Equal(["hunyuan3d-paint", "hunyuan3d-paint21"], MeshModelIds.PaintMesh);
        Assert.All(MeshModelIds.PaintMesh, id => Assert.Contains(id, MeshModelIds.All));
        Assert.All(MeshModelIds.PaintMesh, id => Assert.DoesNotContain(id, MeshModelIds.ImageToMesh));
        Assert.Equal("hunyuan3d-paint", ModelResolver.Resolve("hunyuan3d-paint", null, Modality.Mesh).Catalog?.Id);
    }
}
