using HartsyInference.Audio.Cache;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>AudioModelCache tests. The cache root and directory math is easy to
/// validate offline; the actual download path requires network so we only exercise
/// it in an opt-in test gated on an env var to keep CI fast.</summary>
public sealed class AudioModelCacheTests
{
    [Fact]
    public void GetRepoDirectory_NormalizesSlashToDoubleDash()
    {
        // HF convention: "openai/whisper-tiny" → "openai--whisper-tiny" so a user
        // can symlink an existing HF cache without rename.
        string dir = AudioModelCache.GetRepoDirectory("openai/whisper-tiny");
        Assert.EndsWith("openai--whisper-tiny", dir);
        Assert.True(Directory.Exists(dir));
    }

    [Fact]
    public void CacheRoot_IsAbsolute()
    {
        Assert.True(Path.IsPathRooted(AudioModelCache.CacheRoot));
    }

    [Fact(Skip = "Network test — run manually with: dotnet test --filter Network=Real")]
    [Trait("Network", "Real")]
    public async Task GetAsync_DownloadsFile_FromHuggingFace()
    {
        // openai/whisper-tiny/config.json is small (~2 KB) — safe to pull in a unit test
        // when network is available. Re-running uses the cached copy.
        string path = await AudioModelCache.GetAsync("openai/whisper-tiny", "config.json");
        Assert.True(File.Exists(path));
        string content = await File.ReadAllTextAsync(path);
        Assert.Contains("\"model_type\"", content);
        Assert.Contains("whisper", content, StringComparison.OrdinalIgnoreCase);
    }
}
