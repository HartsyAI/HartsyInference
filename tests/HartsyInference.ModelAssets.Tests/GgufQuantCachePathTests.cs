using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Where the converted-weight cache goes, and the opt-in parallel write. No weights.</summary>
public sealed class GgufQuantCachePathTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-cachepath-" + Guid.NewGuid().ToString("N"));

    public GgufQuantCachePathTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void AWritableFolderKeepsItsCacheBesideTheModel()
    {
        string path = GgufQuantCache.DefaultPath(_root, "a.gguf");
        Assert.Equal(Path.Combine(_root, "_gguf-cache", "a.gguf"), path);
        Assert.Empty(Directory.GetFiles(Path.Combine(_root, "_gguf-cache")));   // the write probe is cleaned up
    }

    [Fact]
    public void AHuggingFaceSnapshotUsesTheCentralCache()
    {
        string snapshot = Path.Combine(_root, "models--x--y", "snapshots", "abc123");
        Directory.CreateDirectory(snapshot);
        string path = GgufQuantCache.DefaultPath(snapshot, "a.gguf");
        Assert.DoesNotContain("snapshots", path);
        Assert.Contains(Path.Combine("hartsyinference", "quant-cache"), path);
        Assert.False(Directory.Exists(Path.Combine(snapshot, "_gguf-cache")));
        // The same folder maps to the same place; another folder does not.
        Assert.Equal(path, GgufQuantCache.DefaultPath(snapshot, "a.gguf"));
        string other = Path.Combine(_root, "models--x--z", "snapshots", "abc123");
        Assert.NotEqual(Path.GetDirectoryName(path), Path.GetDirectoryName(GgufQuantCache.DefaultPath(other, "a.gguf")));
    }

    [Fact]
    public void AFolderThatCannotBeWrittenFallsBackToTheCentralCache()
    {
        // A file where the cache folder would go makes it impossible to create on every platform.
        string blocked = Path.Combine(_root, "blocked");
        Directory.CreateDirectory(blocked);
        File.WriteAllText(Path.Combine(blocked, "_gguf-cache"), "not a folder");
        Assert.Contains(Path.Combine("hartsyinference", "quant-cache"), GgufQuantCache.DefaultPath(blocked, "a.gguf"));
    }

    [Fact]
    public void TheParallelProduceAheadIsBounded() =>
        Assert.InRange(GgufQuantCache.ParallelProduceAhead, 1, 4);
}
