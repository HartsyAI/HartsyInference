using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The file inventory and prefetch of 3D models over a temp models root: nothing is downloaded, because every file a prefetch would fetch is placed first.</summary>
[Collection("ModelsRootKnob")]
public sealed class MeshAssetTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-mesh-assets-" + Guid.NewGuid().ToString("N"));
    private readonly string? _token = Environment.GetEnvironmentVariable("HF_TOKEN");

    public MeshAssetTests()
    {
        Directory.CreateDirectory(_root);
        KnobStore.Set(EngineKnobs.ModelsRoot, _root);
        Environment.SetEnvironmentVariable("HF_TOKEN", null);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("HF_TOKEN", _token);
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        Directory.Delete(_root, recursive: true);
    }

    private static ModelSpec Spec(string id, string? localPath = null) => new()
    {
        Requested = id,
        Modality = Modality.Mesh,
        Catalog = ModelCatalog.Find(id),
        LocalPath = localPath,
    };

    private static void Place(string path, int size = 3)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, new byte[size]);
    }

    private static void PlaceCatalog(string id, Func<ModelAsset, bool>? skip = null)
    {
        foreach (ModelAsset asset in ModelCatalog.Find(id)!.Assets.Where(a => skip?.Invoke(a) != true))
            Place(ModelDownloader.TargetPath(asset));
    }

    [Fact]
    public void AnEmptyRootReportsEveryCatalogFileMissingAndIncomplete()
    {
        MeshAssetStatus status = new MeshAssetService().GetStatus(Spec("hunyuan3d"));

        Assert.False(status.Complete);
        Assert.All(status.Assets, a => Assert.False(a.Present));
        Assert.Equal(2, status.Assets.Count);
        Assert.Equal("3D/hunyuan3d/model.fp16.safetensors", status.Assets[0].RelativePath);
    }

    [Fact]
    public void PlacedFilesAreFoundWithTheirSize()
    {
        PlaceCatalog("hunyuan3d");

        MeshAssetStatus status = new MeshAssetService().GetStatus(Spec("hunyuan3d"));

        Assert.True(status.Complete);
        Assert.All(status.Assets, a => Assert.Equal(3, a.SizeBytes));
        Assert.Empty(status.Missing);
    }

    [Fact]
    public void AGatedFileNeedsATokenAndPixal3DsNafFileNeedsManualPlacement()
    {
        PlaceCatalog("pixal3d", a => a.Role.Contains("gated"));

        MeshAssetInfo dino = new MeshAssetService().GetStatus(Spec("pixal3d")).Assets.Single(a => a.Gated);
        MeshAssetInfo naf = new MeshAssetService().GetStatus(Spec("pixal3d")).Assets.Single(a => a.RelativePath.EndsWith("naf_release.pth"));

        Assert.False(dino.Present);
        Assert.Equal(MeshAssetBlockerKind.Gated, dino.Blocker);
        Assert.Contains("HF_TOKEN", dino.Instructions);
        Assert.Equal(MeshAssetBlockerKind.ManualPlacement, naf.Blocker);
        Assert.True(naf.Required);

        Environment.SetEnvironmentVariable("HF_TOKEN", "hf_test");
        Assert.Null(new MeshAssetService().GetStatus(Spec("pixal3d")).Assets.Single(a => a.Gated).Blocker);
    }

    [Fact]
    public void TheOptionalFilesDoNotMakeTheModelIncomplete()
    {
        PlaceCatalog("pixal3d", a => a.Role.Contains("optional"));
        Place(Path.Combine(_root, "3D", "pixal3d", "naf_release.pth"));

        MeshAssetStatus status = new MeshAssetService().GetStatus(Spec("pixal3d"));

        Assert.True(status.Complete);
        Assert.Contains(status.Missing, a => !a.Required);
    }

    [Fact]
    public void TheSharedFilesAreFoundWhereTheLoadersAlsoLook()
    {
        PlaceCatalog("trellis2", a => a.FileName is "shape_dec_next_dc_f16c32_fp16.safetensors" or "model.safetensors" && !a.Role.Contains("matting"));
        Place(Path.Combine(_root, "TRELLIS2-ShapeVAE", "shape_dec_next_dc_f16c32_fp16.safetensors"));
        Place(Path.Combine(_root, "DINOv3", "model.safetensors"));

        MeshAssetStatus status = new MeshAssetService().GetStatus(Spec("trellis2"));

        Assert.All(status.Assets.Where(a => a.Role is "shape VAE decoder" or "encoder (gated)"), a => Assert.True(a.Present, a.Role));
    }

    [Fact]
    public void Sam3DFindsMetasCkptFilesInTheModelFolder()
    {
        string folder = Path.Combine(_root, "anywhere");
        foreach (string stem in new[] { "ss_generator", "ss_decoder", "slat_generator", "slat_decoder_gs" })
            Place(Path.Combine(folder, stem + ".ckpt"));

        MeshAssetStatus status = new MeshAssetService().GetStatus(Spec("sam3d-objects", folder));

        Assert.Equal(4, status.Assets.Count(a => a.Present));
        Assert.False(status.Complete);
    }

    [Fact]
    public void AModelWithoutCatalogFilesHasAModelBlockerAndNeverCountsAsInstalled()
    {
        MeshAssetStatus status = new MeshAssetService().GetStatus(Spec("triposr"));

        Assert.Empty(status.Assets);
        Assert.Equal(MeshAssetBlockerKind.NoCatalogEntry, status.ModelBlocker!.Value.Kind);
        Assert.False(status.Complete);
        Assert.Throws<NotSupportedException>(() => new MeshAssetService().GetStatus(Spec("hunyuan3d-mv")));
    }

    [Fact]
    public async Task PrefetchOfAModelWithoutCatalogFilesIsUnsupportedWithATypedReason()
    {
        using InferenceEngine engine = new("cpu");

        ModelPrefetchResult result = await engine.ModelPrefetch.PrefetchAsync(Spec("triposr"));

        Assert.False(result.Supported);
        Assert.Equal(MeshAssetBlockerKind.NoCatalogEntry, Assert.Single(result.Blockers).Kind);
    }

    [Fact]
    public async Task PrefetchFetchesNothingWhenEverythingFetchableIsPresentAndReturnsTheBlockersInstead()
    {
        PlaceCatalog("pixal3d");
        using InferenceEngine engine = new("cpu");

        ModelPrefetchResult result = await engine.ModelPrefetch.PrefetchAsync(Spec("pixal3d"));

        Assert.True(result.Supported);
        MeshAssetBlocker blocker = Assert.Single(result.Blockers);
        Assert.Equal(MeshAssetBlockerKind.ManualPlacement, blocker.Kind);
        Assert.Contains("NAF", blocker.Role);
        Assert.NotNull(result.PrimaryPath);
    }

    /// <summary>pixal3d with everything placed except the gated encoder and the primary file, and a token set so both are fetchable.</summary>
    private static ModelSpec PixalMissingEncoderAndPrimary(string root)
    {
        Environment.SetEnvironmentVariable("HF_TOKEN", "hf_test");
        PlaceCatalog("pixal3d", a => a.Role.Contains("gated") || a.Role == "config");
        Place(Path.Combine(root, "3D", "pixal3d", "naf_release.pth"));
        return Spec("pixal3d");
    }

    [Fact]
    public async Task AFileTheServerRefusesBecomesAGatedBlockerAndTheRestStillDownloadsWithThePrimaryLast()
    {
        ModelSpec spec = PixalMissingEncoderAndPrimary(_root);
        List<string> order = [];

        ModelPrefetchResult result = await ModelPrefetchService.PrefetchMeshAsync(spec, null, (asset, _) =>
        {
            order.Add(asset.Role);
            if (asset.Role.Contains("gated"))
                throw new HartsyInference.Core.Exceptions.HartsyInferenceException("HuggingFace API request failed for download: HTTP 403 Forbidden. Body: ");
            Place(ModelDownloader.TargetPath(asset));
            return Task.CompletedTask;
        }, CancellationToken.None);

        Assert.Equal(["encoder (gated)", "config"], order);
        Assert.True(result.Supported);
        MeshAssetBlocker blocker = Assert.Single(result.Blockers);
        Assert.Equal(MeshAssetBlockerKind.Gated, blocker.Kind);
        Assert.Contains("HF_TOKEN", blocker.Message);
        Assert.NotNull(result.PrimaryPath);
    }

    [Theory]
    [InlineData(typeof(HttpRequestException))]
    [InlineData(typeof(IOException))]
    public async Task ANetworkFailureBecomesAFetchFailedBlockerNotAnException(Type failure)
    {
        ModelSpec spec = PixalMissingEncoderAndPrimary(_root);

        ModelPrefetchResult result = await ModelPrefetchService.PrefetchMeshAsync(spec, null,
            (_, _) => throw (Exception)Activator.CreateInstance(failure, "connection reset")!, CancellationToken.None);

        Assert.Equal(2, result.Blockers.Count);
        Assert.All(result.Blockers, b => Assert.Equal(MeshAssetBlockerKind.FetchFailed, b.Kind));
        Assert.Null(result.PrimaryPath);
    }

    [Fact]
    public async Task ACancellationStillPropagates()
    {
        ModelSpec spec = PixalMissingEncoderAndPrimary(_root);

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            ModelPrefetchService.PrefetchMeshAsync(spec, null, (_, _) => throw new OperationCanceledException(), CancellationToken.None));
    }

    [Fact]
    public void TheManualFileIsLookedUpBesideTheModelFolderTheCallerNamed()
    {
        string custom = Path.Combine(_root, "elsewhere");
        PlaceCatalog("pixal3d");
        Place(Path.Combine(_root, "3D", "pixal3d", "naf_release.pth"));

        MeshAssetInfo naf = new MeshAssetService().GetStatus(Spec("pixal3d", custom)).Assets.Single(a => a.RelativePath.EndsWith("naf_release.pth"));
        Assert.False(naf.Present);
        Place(Path.Combine(custom, "naf_release.pth"));
        MeshAssetInfo placed = new MeshAssetService().GetStatus(Spec("pixal3d", custom)).Assets.Single(a => a.RelativePath.EndsWith("naf_release.pth"));

        Assert.True(placed.Present);
    }

    [Fact]
    public async Task FilesFoundInTheLoadersOtherFoldersAreListedAsInstalled()
    {
        PlaceCatalog("trellis2", a => a.FileName == "shape_dec_next_dc_f16c32_fp16.safetensors");
        Place(Path.Combine(_root, "TRELLIS2-ShapeVAE", "shape_dec_next_dc_f16c32_fp16.safetensors"));
        Environment.SetEnvironmentVariable("HF_TOKEN", "hf_test");

        ModelPrefetchResult result = await ModelPrefetchService.PrefetchMeshAsync(Spec("trellis2"), null, (_, _) => Task.CompletedTask, CancellationToken.None);

        Assert.Contains(result.Files, f => f.Contains("TRELLIS2-ShapeVAE"));
    }

    [Fact]
    public async Task ThePlanOfAModelWithoutCatalogFilesRepeatsTheBlockersAdvice()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("triposr"), new Requests.MeshRequest { Image = new Requests.ImageData { Rgb = new byte[12], Width = 2, Height = 2 } });

        MeshPlanIssue issue = Assert.Single(plan.Issues, i => i.Code == "mesh.model.not_installed");
        Assert.Contains("checkpoint folder", issue.Message);
    }

    [Fact]
    public async Task APlanListsTheMissingFilesAsWarningsAndCarriesTheInventory()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("hunyuan3d", _root), new Requests.MeshRequest { Image = new Requests.ImageData { Rgb = new byte[12], Width = 2, Height = 2 } });

        Assert.True(plan.IsValid);
        Assert.Equal(2, plan.Assets.Missing.Count());
        Assert.Equal(2, plan.Issues.Count(i => i.Code == "mesh.artifact.missing"));
    }
}
