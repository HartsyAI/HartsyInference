using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for <see cref="HeartMulaMusicModel.ResolveQuantCachePath(string, string, string, string, Func{string, bool})"/>:
/// resolves under the shared audio root by default, the legacy path only when a file already exists
/// there.</summary>
public sealed class HeartMulaQuantCachePathTests
{
    private const string LegacyRoot = "/fake/home/.cache/hartsyinference";
    private const string SharedAudioRoot = "/fake/models/audio";

    [Fact]
    public void NoLegacyFile_ResolvesUnderTheSharedAudioRoot_NotTheLegacyOne()
    {
        string path = HeartMulaMusicModel.ResolveQuantCachePath(
            "HeartMuLa/HeartMuLa-oss-3B", "q8_0", LegacyRoot, SharedAudioRoot, fileExists: _ => false);

        Assert.Equal(
            Path.Combine(SharedAudioRoot, "music", "heartmula", "HeartMuLa_HeartMuLa-oss-3B-q8_0.gguf"),
            path);
    }

    [Fact]
    public void LegacyFileAlreadyExists_ResolvesToTheLegacyPath_NotTheSharedRoot()
    {
        string expectedLegacyPath = Path.Combine(LegacyRoot, "heartmula", "HeartMuLa_HeartMuLa-oss-3B-q8_0.gguf");

        string path = HeartMulaMusicModel.ResolveQuantCachePath(
            "HeartMuLa/HeartMuLa-oss-3B", "q8_0", LegacyRoot, SharedAudioRoot,
            fileExists: candidate => candidate == expectedLegacyPath);

        Assert.Equal(expectedLegacyPath, path);
    }

    [Fact]
    public void FileExistsCheck_IsAskedAboutTheLegacyPath_NotSomeOtherPath()
    {
        string? asked = null;
        HeartMulaMusicModel.ResolveQuantCachePath(
            "HeartMuLa/HeartMuLa-oss-3B", "q4_k", LegacyRoot, SharedAudioRoot,
            fileExists: candidate => { asked = candidate; return false; });

        Assert.Equal(Path.Combine(LegacyRoot, "heartmula", "HeartMuLa_HeartMuLa-oss-3B-q4_k.gguf"), asked);
    }

    [Theory]
    [InlineData("HeartMuLa/HeartMuLa-oss-3B", "q8_0", "HeartMuLa_HeartMuLa-oss-3B-q8_0.gguf")]
    [InlineData("HeartMuLa/HeartMuLa-RL-oss-3B-20260123", "q4_k", "HeartMuLa_HeartMuLa-RL-oss-3B-20260123-q4_k.gguf")]
    [InlineData("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "Q8_0", "HeartMuLa_HeartMuLa-oss-3B-happy-new-year-q8_0.gguf")]
    public void FileName_ReplacesRepoSlashWithUnderscore_AndLowercasesTheQuantSuffix(string repo, string quant, string expectedFileName)
    {
        string path = HeartMulaMusicModel.ResolveQuantCachePath(repo, quant, LegacyRoot, SharedAudioRoot, fileExists: _ => false);

        Assert.Equal(Path.Combine(SharedAudioRoot, "music", "heartmula", expectedFileName), path);
    }

    [Fact]
    public void RealOverload_ResolvesUnderAudioModelCacheCacheRoot_WhenNoLegacyFileIsPresent()
    {
        // Exercises the real two-argument overload (real AudioModelCache.CacheRoot, real home directory,
        // real File.Exists) -- doesn't assert an exact root, since that's environment-dependent (ModelsRoot/
        // ModelCacheRoot may or may not be configured in the test environment), only that the result is
        // rooted under whatever AudioModelCache.CacheRoot actually resolves to right now, under
        // music/heartmula, with the expected file name -- unless a real legacy file genuinely exists on this
        // machine, which the test environment should not have.
        string repo = "HeartMuLa/HeartMuLa-oss-3B";
        string fileName = "HeartMuLa_HeartMuLa-oss-3B-q8_0.gguf";
        string legacyPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "hartsyinference", "heartmula", fileName);

        string path = HeartMulaMusicModel.ResolveQuantCachePath(repo, "q8_0");

        if (File.Exists(legacyPath))
        {
            Assert.Equal(legacyPath, path);
        }
        else
        {
            Assert.Equal(
                Path.Combine(HartsyInference.Audio.Cache.AudioModelCache.CacheRoot, "music", "heartmula", fileName),
                path);
        }
    }
}
