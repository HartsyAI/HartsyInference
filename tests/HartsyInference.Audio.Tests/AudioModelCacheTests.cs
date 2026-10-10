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
        string dir = AudioModelCache.GetRepoDirectory("openai/whisper-tiny", "stt");
        Assert.EndsWith("openai--whisper-tiny", dir);
        Assert.True(Directory.Exists(dir));
    }

    [Fact(Skip = "Network test — run manually with: dotnet test --filter Network=Real")]
    [Trait("Network", "Real")]
    public async Task FetchAllAsync_ResolvesEveryRequiredFile_AndToleratesMissingOptional()
    {
        // whisper-tiny is English+multilingual and ~150 MB; "nonexistent.json" proves an absent optional file
        // does not fail the fetch, which is what lets one file list serve variants that ship different extras.
        List<AudioModelFile> files =
        [
            new("config.json"),
            new("nonexistent-optional.json", Required: false),
            new("vocab.json"),
        ];
        IReadOnlyDictionary<string, string> fetched = await AudioModelCache.FetchAllAsync("openai/whisper-tiny", files, "stt");
        Assert.True(File.Exists(fetched["config.json"]));
        Assert.True(File.Exists(fetched["vocab.json"]));
        Assert.False(fetched.ContainsKey("nonexistent-optional.json"));
    }

    [Fact(Skip = "Network test — run manually with: dotnet test --filter Network=Real")]
    [Trait("Network", "Real")]
    public async Task FetchAllAsync_ThrowsWhenARequiredFileIsMissing()
    {
        List<AudioModelFile> files = [new("config.json"), new("definitely-not-a-real-file.bin")];
        await Assert.ThrowsAnyAsync<Exception>(() => AudioModelCache.FetchAllAsync("openai/whisper-tiny", files, "stt"));
    }
}
