using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Covers <see cref="ModelDownloader.TargetPath"/>'s legacy-name fallback. Renaming a side model to match
/// what SwarmUI downloads is only a win if an install that already holds the file under the OLD name keeps using
/// it — otherwise the rename costs every upgrading install a multi-gigabyte re-download, and the recipes that
/// resolve shared assets through the strict non-downloading overload (Mage-Flow shares Krea 2's encoder) report
/// the encoder missing instead. Points <c>RepoPaths.ModelsRoot()</c> at a hermetic fixture; the override is
/// process-global, so it is cleared after each test.</summary>
public sealed class SideModelLegacyPathTests : IDisposable
{
    private readonly string _tempModelsRoot = Path.Combine(Path.GetTempPath(), "hartsy-legacypath-tests-" + Path.GetRandomFileName());

    public SideModelLegacyPathTests()
    {
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
    public void TargetPath_PrefersTheCanonicalName_WhenItIsPresent()
    {
        string canonical = Place(SideModels.Qwen3VL_4B.TargetSubdir, SideModels.Qwen3VL_4B.FileName);
        Place(SideModels.Qwen3VL_4B.TargetSubdir, SideModels.Qwen3VL_4B.LegacyTargetNames[0]);
        Assert.Equal(canonical, ModelDownloader.TargetPath(SideModels.Qwen3VL_4B));
    }

    [Fact]
    public void TargetPath_FallsBackToALegacyName_SoAnUpgradeDoesNotRefetch()
    {
        string legacy = Place(SideModels.Qwen3VL_4B.TargetSubdir, SideModels.Qwen3VL_4B.LegacyTargetNames[0]);
        Assert.Equal(legacy, ModelDownloader.TargetPath(SideModels.Qwen3VL_4B));
    }

    [Fact]
    public void TargetPath_ReturnsTheCanonicalName_WhenNothingIsOnDisk()
    {
        // A fresh install must still download to the canonical (SwarmUI-matching) name, not to a legacy one.
        string expected = Path.Combine(_tempModelsRoot, SideModels.Qwen3VL_4B.TargetSubdir, SideModels.Qwen3VL_4B.FileName);
        Assert.Equal(expected, ModelDownloader.TargetPath(SideModels.Qwen3VL_4B));
    }

    [Fact]
    public void TargetPath_IsUnaffectedForAnAssetThatNeverMoved()
    {
        Assert.Empty(SideModels.QwenImageVae.LegacyTargetNames);
        string expected = Path.Combine(_tempModelsRoot, SideModels.QwenImageVae.TargetSubdir, SideModels.QwenImageVae.FileName);
        Assert.Equal(expected, ModelDownloader.TargetPath(SideModels.QwenImageVae));
    }

    private string Place(string subdir, string relativeName)
    {
        string path = Path.Combine(_tempModelsRoot, subdir, relativeName);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [0x00]);
        return path;
    }
}
