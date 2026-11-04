using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Covers the SeedVR2-7B transformer's legacy-name fallback. The catalog asks for the numz file name, but an
/// install that already holds the same bytes under the older name must keep using it instead of downloading 16 GB
/// again. Points <c>RepoPaths.ModelsRoot()</c> at a hermetic fixture; the override is process-global, so it is cleared
/// after each test.</summary>
[Collection("ModelsRootKnob")]
public sealed class SeedVr2LegacyPathTests : IDisposable
{
    private readonly string _tempModelsRoot = Path.Combine(Path.GetTempPath(), "hartsy-seedvr2-legacypath-tests-" + Path.GetRandomFileName());

    public SeedVr2LegacyPathTests()
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
    public void TargetPath_UsesTheInstalledLegacyFile_SoNothingIsRedownloaded()
    {
        string legacy = Place("Video/SeedVr2", "seedvr2_7b_dit_f16.safetensors");
        Assert.Equal(legacy, ModelDownloader.TargetPath(Transformer7B()));
    }

    [Fact]
    public void MissingAssets_DoesNotListTheTransformer_WhenTheLegacyFileIsInstalled()
    {
        Place("Video/SeedVr2", "seedvr2_7b_dit_f16.safetensors");
        Assert.DoesNotContain(ModelDownloader.MissingAssets(ModelCatalog.Find("seedvr2-7b")!), asset => asset.Role == "transformer");
    }

    [Fact]
    public void TargetPath_FallsBackToTheCanonicalName_WhenNothingIsOnDisk()
    {
        // A fresh install must still download to the canonical name, not the legacy one.
        string expected = Path.Combine(_tempModelsRoot, "Video/SeedVr2", "seedvr2_ema_7b_fp16.safetensors");
        Assert.Equal(expected, ModelDownloader.TargetPath(Transformer7B()));
    }

    private static ModelAsset Transformer7B() =>
        ModelCatalog.Find("seedvr2-7b")!.Assets.Single(asset => asset.Role == "transformer");

    private string Place(string subdir, string relativeName)
    {
        string path = Path.Combine(_tempModelsRoot, subdir, relativeName);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [0x00]);
        return path;
    }
}
