using System.Text.Json;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Locks the rule that generated media is written to disk by default: any CLI or API call lands in the
/// shared output root unless the request names a different directory or opts out. Before this the HTTP API wrote
/// nothing at all — every artifact existed only as base64 in a response body.</summary>
public sealed class ArtifactPersistenceTests
{
    /// <summary>Concrete stand-in for the abstract envelope; the persistence layer only reads its two fields.</summary>
    private sealed class TestRequest : NativeArtifactRequest;

    [Fact]
    public void ResolveDir_Unset_IsTheSharedOutputRoot()
    {
        Assert.Equal(RepoPaths.OutputRoot(), OutputWriter.ResolveDir(null));
        Assert.Equal(RepoPaths.OutputRoot(), OutputWriter.ResolveDir("   "));
    }

    [Fact]
    public void Save_WritesIntoTheRequestedDirectory_AndAutoNumbers()
    {
        string dir = NewTempDir();
        try
        {
            TestRequest req = new TestRequest { OutputDir = dir };
            string? first = ArtifactPersistence.Save(req, [1, 2, 3], "a lighthouse keeper", "png");
            string? second = ArtifactPersistence.Save(req, [4, 5, 6], "a lighthouse keeper", "png");

            Assert.NotNull(first);
            Assert.NotNull(second);
            Assert.NotEqual(first, second);
            Assert.Equal("a-lighthouse-keeper-0001.png", Path.GetFileName(first));
            Assert.Equal("a-lighthouse-keeper-0002.png", Path.GetFileName(second));
            Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(first!));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Save_OptedOut_WritesNothing()
    {
        string dir = NewTempDir();
        try
        {
            string? path = ArtifactPersistence.Save(new TestRequest { Save = false, OutputDir = dir }, [1, 2, 3], "p", "png");
            Assert.Null(path);
            Assert.Empty(Directory.GetFileSystemEntries(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Save_EmptyPayload_WritesNothing()
    {
        string dir = NewTempDir();
        try
        {
            Assert.Null(ArtifactPersistence.Save(new TestRequest { OutputDir = dir }, [], "p", "png"));
            Assert.Empty(Directory.GetFileSystemEntries(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void SaveGroup_PutsEveryStemInOneDirectory()
    {
        string dir = NewTempDir();
        try
        {
            Dictionary<string, byte[]> stems = new Dictionary<string, byte[]>
            {
                ["vocals"] = [1],
                ["drums"] = [2],
            };
            string? saved = ArtifactPersistence.SaveGroup(new TestRequest { OutputDir = dir }, stems, "separate", "wav");

            Assert.NotNull(saved);
            Assert.True(Directory.Exists(saved));
            Assert.True(File.Exists(Path.Combine(saved!, "vocals.wav")));
            Assert.True(File.Exists(Path.Combine(saved!, "drums.wav")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>The two controls must arrive from JSON — they are inherited members, and inheritance is exactly
    /// where a binder is most likely to silently drop them.</summary>
    [Fact]
    public void Envelope_BindsSaveAndOutputDirFromJson()
    {
        NativeImageRequest? req = JsonSerializer.Deserialize<NativeImageRequest>(
            """{"model":"m","request":{"prompt":"p"},"save":false,"outputDir":"/tmp/somewhere"}""",
            new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.NotNull(req);
        Assert.False(req!.Save);
        Assert.Equal("/tmp/somewhere", req.OutputDir);
    }

    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"hartsy-artifact-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }
}
