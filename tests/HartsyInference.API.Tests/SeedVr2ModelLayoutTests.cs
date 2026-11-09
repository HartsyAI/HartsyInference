using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Pins where the SeedVR2 files are saved and which file each catalog id runs. The Comfy-Org repacks are the
/// defaults and live in diffusion_models/, where ComfyUI looks for DiTs. The numz originals stay selectable under their
/// own ids and keep their installed names. The VAE goes to VAE/SeedVR2 under the name core registers. Points
/// <c>RepoPaths.ModelsRoot()</c> at a hermetic fixture; the override is process-global, so it is cleared after each test.</summary>
[Collection("ModelsRootKnob")]
public sealed class SeedVr2ModelLayoutTests : IDisposable
{
    private readonly string _tempModelsRoot = Path.Combine(Path.GetTempPath(), "hartsy-seedvr2-layout-tests-" + Path.GetRandomFileName());

    public SeedVr2ModelLayoutTests()
    {
        // Load the settings file first: its first load would overwrite the override set below.
        _ = EngineKnobs.ModelsRoot.Value;
        Directory.CreateDirectory(_tempModelsRoot);
        KnobStore.Set(EngineKnobs.ModelsRoot, _tempModelsRoot);
    }

    public void Dispose()
    {
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        if (Directory.Exists(_tempModelsRoot))
            Directory.Delete(_tempModelsRoot, recursive: true);
    }

    [Fact]
    public void Vae_SavesToTheNameCoreRegisters()
    {
        string expected = Path.Combine(_tempModelsRoot, "VAE", "SeedVR2", "seedvr2_ema_vae_fp16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-3b", "vae")));
    }

    [Fact]
    public void Transformer3B_DefaultIsTheOfficialCheckpointInDiffusionModels()
    {
        string expected = Path.Combine(_tempModelsRoot, "diffusion_models", "seedvr2_ema_3b.pth");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-3b", "transformer")));
        Assert.Equal("ByteDance-Seed/SeedVR2-3B", Asset("seedvr2-3b", "transformer").Repo);
    }

    [Fact]
    public void Transformer3BComfy_IsTheRepackInDiffusionModels()
    {
        string expected = Path.Combine(_tempModelsRoot, "diffusion_models", "seedvr2_3b_fp16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-3b-comfy", "transformer")));
        Assert.Equal("Comfy-Org/SeedVR2", Asset("seedvr2-3b-comfy", "transformer").Repo);
    }

    [Fact]
    public void Transformer7B_DefaultIsTheComfyOrgRepackInDiffusionModels()
    {
        string expected = Path.Combine(_tempModelsRoot, "diffusion_models", "seedvr2_7b_fp16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-7b", "transformer")));
    }

    [Fact]
    public void Transformer7B_ReusesAnInstalledComfyOrgFile_SoNothingIsRedownloaded()
    {
        string installed = Place("diffusion_models", "seedvr2_7b_fp16.safetensors");
        Assert.Equal(installed, ModelDownloader.TargetPath(Asset("seedvr2-7b", "transformer")));
        Assert.DoesNotContain(ModelDownloader.MissingAssets(ModelCatalog.Find("seedvr2-7b")!), asset => asset.Role == "transformer");
    }

    [Fact]
    public void Transformer7BNumz_OldInstallNameIsStillFound_SoNothingIsRedownloaded()
    {
        // Installs from the previous catalog saved the numz 7B transformer under its numz name.
        string installed = Place("Stable-Diffusion/SeedVR2", "seedvr2_ema_7b_fp16.safetensors");
        Assert.Equal(installed, ModelDownloader.TargetPath(Asset("seedvr2-7b-numz", "transformer")));
        Assert.DoesNotContain(ModelDownloader.MissingAssets(ModelCatalog.Find("seedvr2-7b-numz")!), asset => asset.Role == "transformer");
    }

    [Fact]
    public void Transformer3BNumz_LivesInTheStableDiffusionFolderSwarmListsIt()
    {
        string expected = Path.Combine(_tempModelsRoot, "Stable-Diffusion", "SeedVR2", "seedvr2_ema_3b_fp16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-3b-numz", "transformer")));
    }

    [Fact]
    public void Transformer7BNumz_FreshInstallLandsInTheStableDiffusionFolder()
    {
        string expected = Path.Combine(_tempModelsRoot, "Stable-Diffusion", "SeedVR2", "seedvr2_7b_dit_f16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-7b-numz", "transformer")));
    }

    [Fact]
    public void NoVariantTransformerTargetsTheVideoFolder()
    {
        // Swarm has no Video model root for SeedVR. The engine's own embedding is the only SeedVR file left in Video.
        foreach (CatalogEntry entry in ModelCatalog.ForModality(Modality.Restore).Where(e => e.Id.StartsWith("seedvr2-", StringComparison.Ordinal)))
        {
            string subdir = entry.Assets.Single(asset => asset.Role == "transformer").TargetSubdir;
            Assert.DoesNotContain("Video", subdir, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Vae_IsTheSameFileInEveryVariant()
    {
        ModelAsset vae3B = Asset("seedvr2-3b", "vae");
        ModelAsset vae7B = Asset("seedvr2-7b", "vae");
        Assert.Equal(vae3B.FileName, vae7B.FileName);
        Assert.Equal(vae3B.TargetSubdir, vae7B.TargetSubdir);
        Assert.Equal(vae3B.Sha256, vae7B.Sha256);
    }

    [Fact]
    public void EveryVariant_NamesADistinctTransformer()
    {
        // Two ids on one file would run the wrong weights for one of them without any error.
        List<ModelAsset> transformers = ModelCatalog.ForModality(Modality.Restore)
            .Where(entry => entry.Id.StartsWith("seedvr2-", StringComparison.Ordinal))
            .Select(entry => entry.Assets.Single(asset => asset.Role == "transformer"))
            .ToList();
        Assert.Equal(transformers.Count, transformers.Select(asset => asset.RepoPath).Distinct().Count());
        Assert.Equal(transformers.Count, transformers.Select(asset => asset.Sha256).Distinct().Count());
    }

    private static ModelAsset Asset(string catalogId, string role) =>
        ModelCatalog.Find(catalogId)!.Assets.Single(asset => asset.Role == role);

    private string Place(string subdir, string relativeName)
    {
        string path = Path.Combine(_tempModelsRoot, subdir, relativeName);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [0x00]);
        return path;
    }
}
