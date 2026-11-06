using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Pins where the SeedVR2 files are saved. Swarm core loads the VAE from VAE/SeedVR2 under the name core
/// registers, so the engine saves the VAE there. The transformers stay in Video/SeedVr2: core lists a SeedVR2
/// checkpoint from the model folders, and numz's transformers have no conditioning tensors, so core cannot run them
/// from there. Points <c>RepoPaths.ModelsRoot()</c> at a hermetic fixture; the override is process-global, so it is
/// cleared after each test.</summary>
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
    public void Transformer3B_SavesToTheVideoFolderTheEngineReads()
    {
        string expected = Path.Combine(_tempModelsRoot, "Video", "SeedVr2", "seedvr2_ema_3b_fp16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-3b", "transformer")));
    }

    [Fact]
    public void Transformer7B_UsesTheInstalledFile_SoNothingIsRedownloaded()
    {
        string installed = Place("Video/SeedVr2", "seedvr2_7b_dit_f16.safetensors");
        Assert.Equal(installed, ModelDownloader.TargetPath(Asset("seedvr2-7b", "transformer")));
        Assert.DoesNotContain(ModelDownloader.MissingAssets(ModelCatalog.Find("seedvr2-7b")!), asset => asset.Role == "transformer");
    }

    [Fact]
    public void Transformer7B_FreshInstallSavesUnderTheKeptName()
    {
        string expected = Path.Combine(_tempModelsRoot, "Video", "SeedVr2", "seedvr2_7b_dit_f16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Asset("seedvr2-7b", "transformer")));
    }

    [Fact]
    public void Transformer7B_OldInstallNameIsStillFound_SoNothingIsRedownloaded()
    {
        // Installs from the previous catalog saved the transformer under its numz name.
        string installed = Place("Video/SeedVr2", "seedvr2_ema_7b_fp16.safetensors");
        Assert.Equal(installed, ModelDownloader.TargetPath(Asset("seedvr2-7b", "transformer")));
        Assert.DoesNotContain(ModelDownloader.MissingAssets(ModelCatalog.Find("seedvr2-7b")!), asset => asset.Role == "transformer");
    }

    [Fact]
    public void Vae_IsTheSameFileInBothEntries()
    {
        ModelAsset vae3B = Asset("seedvr2-3b", "vae");
        ModelAsset vae7B = Asset("seedvr2-7b", "vae");
        Assert.Equal(vae3B.FileName, vae7B.FileName);
        Assert.Equal(vae3B.TargetSubdir, vae7B.TargetSubdir);
        Assert.Equal(vae3B.Sha256, vae7B.Sha256);
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
