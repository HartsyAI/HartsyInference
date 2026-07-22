using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Direct tests for <see cref="ModelResolver"/>'s catalog-id resolution — specifically the fix for a real
/// bug found via a live end-to-end run: several image/video catalog entries record their actual on-disk location
/// via <c>Assets[].TargetSubdir</c> (e.g. "Stable-Diffusion/Krea2"), which does not match the coarse
/// per-modality subdir guess ("Image"/"Video") <see cref="ModelResolver"/> falls back to — so a bare catalog id
/// alone used to 400 with "no checkpoint found" even when the file was genuinely present. Uses the
/// <c>HARTSYINFERENCE_MODELS</c> env override (honored by <c>RepoPaths.ModelsRoot()</c>) for a hermetic fixture —
/// no real multi-GB checkpoint needed. Not parallelized against other test classes that might read the same
/// process-global env var; restores it immediately after each test to minimize the exposure window.</summary>
public sealed class ModelResolverTests : IDisposable
{
    private readonly string _tempModelsRoot = Path.Combine(Path.GetTempPath(), "hartsy-modelresolver-tests-" + Path.GetRandomFileName());
    private readonly string? _previousOverride = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS");

    public ModelResolverTests()
    {
        Directory.CreateDirectory(_tempModelsRoot);
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_MODELS", _tempModelsRoot);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("HARTSYINFERENCE_MODELS", _previousOverride);
        if (Directory.Exists(_tempModelsRoot))
            Directory.Delete(_tempModelsRoot, recursive: true);
    }

    [Fact]
    public void Resolve_PrefersAuthoritativeAssetPath_OverNaiveSubdirGuess()
    {
        // krea2's transformer asset: TargetSubdir "Stable-Diffusion/Krea2", file "krea2_turbo_fp8_scaled.safetensors"
        // — NOT under the naive per-modality guess "Image/krea2" this bug used to fall back to.
        string authoritativeDir = Path.Combine(_tempModelsRoot, "Stable-Diffusion", "Krea2");
        Directory.CreateDirectory(authoritativeDir);
        string authoritativeFile = Path.Combine(authoritativeDir, "krea2_turbo_fp8_scaled.safetensors");
        File.WriteAllBytes(authoritativeFile, [0x01]);

        ModelSpec spec = ModelResolver.Resolve("krea2", modelPathArg: null, Modality.Image);

        Assert.Equal(Path.GetFullPath(authoritativeFile), spec.LocalPath);
    }

    [Fact]
    public void Resolve_MissingAsset_ReturnsNullLocalPath_NotAThrow()
    {
        // No files created under _tempModelsRoot at all -- should resolve to null (caller/Engine surfaces the
        // "no checkpoint" error), not throw or silently pick a wrong path.
        ModelSpec spec = ModelResolver.Resolve("krea2", modelPathArg: null, Modality.Image);

        Assert.Null(spec.LocalPath);
        Assert.NotNull(spec.Catalog);
        Assert.Equal("krea2", spec.Catalog!.Id);
    }

    [Fact]
    public void Resolve_EntryWithNoAssets_StillFallsBackToLegacySubdirGuess()
    {
        // "qwen2" is catalogued with Assets = [] (empty) -- must still fall back to the legacy
        // Models/<ModalitySubdir>/<id> convention for manually-placed checkpoints, unaffected by this fix.
        string legacyDir = Path.Combine(_tempModelsRoot, "LLM", "qwen2");
        Directory.CreateDirectory(legacyDir);
        string legacyFile = Path.Combine(legacyDir, "model.gguf");
        File.WriteAllBytes(legacyFile, [0x01]);

        ModelSpec spec = ModelResolver.Resolve("qwen2", modelPathArg: null, Modality.Text);

        Assert.Equal(Path.GetFullPath(legacyFile), spec.LocalPath);
    }

    [Fact]
    public void Resolve_ExplicitModelPath_StillWinsOverCatalogAssetPath()
    {
        string authoritativeDir = Path.Combine(_tempModelsRoot, "Stable-Diffusion", "Krea2");
        Directory.CreateDirectory(authoritativeDir);
        File.WriteAllBytes(Path.Combine(authoritativeDir, "krea2_turbo_fp8_scaled.safetensors"), [0x01]);

        string overridePath = Path.Combine(_tempModelsRoot, "custom-checkpoint.safetensors");
        File.WriteAllBytes(overridePath, [0x02]);

        ModelSpec spec = ModelResolver.Resolve("krea2", overridePath, Modality.Image);

        Assert.Equal(Path.GetFullPath(overridePath), spec.LocalPath);
    }
}
