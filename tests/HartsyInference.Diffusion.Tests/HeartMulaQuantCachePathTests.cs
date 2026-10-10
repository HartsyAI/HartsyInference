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

    [Theory]
    [InlineData("HeartMuLa/HeartMuLa-oss-3B", "q8_0", "HeartMuLa_HeartMuLa-oss-3B-q8_0.gguf")]
    [InlineData("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", "Q8_0", "HeartMuLa_HeartMuLa-oss-3B-happy-new-year-q8_0.gguf")]
    public void FileName_ReplacesRepoSlashWithUnderscore_AndLowercasesTheQuantSuffix(string repo, string quant, string expectedFileName)
    {
        string path = HeartMulaMusicModel.ResolveQuantCachePath(repo, quant, LegacyRoot, SharedAudioRoot, fileExists: _ => false);

        Assert.Equal(Path.Combine(SharedAudioRoot, "music", "heartmula", expectedFileName), path);
    }
}
