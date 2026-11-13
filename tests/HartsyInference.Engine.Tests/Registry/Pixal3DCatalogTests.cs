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
        // pipeline.json, three flows with their configs, the two decoders, the optional texture flow with its config and decoder, and NAF.
        Assert.Equal(13, entry.Assets.Count(a => a.TargetSubdir == "3D/pixal3d"));
        Assert.Contains(entry.Assets, a => a.RepoPath == "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors" && a.Repo == "TencentARC/Pixal3D");
    }

    [Fact]
    public void Entry_FetchesNafFromItsGitHubReleaseWithThePinnedHashAndListsItsMirrorRepack()
    {
        ModelAsset naf = Assert.Single(ModelCatalog.Find("pixal3d")!.Assets, a => a.FileName == "naf_release.pth");

        Assert.Equal("https://github.com/valeoai/NAF/releases/download/model/naf_release.pth", naf.Url);
        Assert.Equal("c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f", naf.Sha256);
        Assert.Equal("3D/pixal3d", naf.TargetSubdir);
        Assert.Equal("pixal3d/naf_release.safetensors", naf.Repack!.MirrorPath);
        Assert.False(MeshAssetService.IsOptional(naf));
    }

    [Fact]
    public void Entry_CarriesTheTextureFlowAndDecoderWithTheirLfsHashes()
    {
        CatalogEntry entry = ModelCatalog.Find("pixal3d")!;

        ModelAsset flow = Assert.Single(entry.Assets, a => a.RepoPath == "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors");
        Assert.Equal("65df179ad02f9e92bc18335d7861fadd95f268e6249217352468dd83cb79d1d0", flow.Sha256);
        Assert.Contains(entry.Assets, a => a.RepoPath == "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json");
        ModelAsset decoder = Assert.Single(entry.Assets, a => a.RepoPath == "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors");
        Assert.Equal("97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c", decoder.Sha256);
        Assert.All([flow, decoder], asset => Assert.Equal("3D/pixal3d", asset.TargetSubdir));
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
