using System.Security.Cryptography;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Features;
using Xunit;

namespace HartsyInference.Engine.Tests.Features;

/// <summary>A real file on disk at a pinned side-model path goes through <see cref="ModelFileLocator.RequireOrEnsure"/>: a file
/// whose SHA-256 matches loads; a same-named file from another build is refused. The models root is a temp folder, set through the
/// knob store and cleared after each test (the override is process-global).</summary>
[Collection("ModelsRootKnob")]
public sealed class RequireOrEnsurePinTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-pin-tests-" + Path.GetRandomFileName());

    public RequireOrEnsurePinTests()
    {
        // Load the settings file first: its first load would overwrite the override set below.
        _ = EngineKnobs.ModelsRoot.Value;
        Directory.CreateDirectory(_root);
        KnobStore.Set(EngineKnobs.ModelsRoot, _root);
    }

    public void Dispose()
    {
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    private static ModelAsset Asset(string sha256) => new()
    {
        Repo = "test/repo",
        RepoPath = "text_encoders/test.safetensors",
        TargetSubdir = "text_encoders",
        TargetName = "fibo2/pin-test/test.safetensors",
        Role = "text encoder",
        Sha256 = sha256,
    };

    [Fact]
    public void FileWithTheWrongHash_IsRefused_AndNotLoaded()
    {
        ModelAsset pinned = Asset(Sha256Of([1, 2, 3]));
        string target = ModelDownloader.TargetPath(pinned);
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        File.WriteAllBytes(target, [9, 9, 9, 9]); // a same-named file from another build

        InvalidDataException ex = Assert.Throws<InvalidDataException>(() =>
            ModelFileLocator.RequireOrEnsure(null, pinned, "text encoder", CancellationToken.None));

        Assert.Contains(pinned.Sha256!, ex.Message);
        Assert.Contains("Refusing to load it", ex.Message);
        Assert.Equal([9, 9, 9, 9], File.ReadAllBytes(target)); // the file is left as it was
    }

    [Fact]
    public void FileWithThePinnedHash_IsReturned()
    {
        byte[] bytes = [1, 2, 3];
        ModelAsset pinned = Asset(Sha256Of(bytes));
        string target = ModelDownloader.TargetPath(pinned);
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        File.WriteAllBytes(target, bytes);

        string path = ModelFileLocator.RequireOrEnsure(null, pinned, "text encoder", CancellationToken.None);

        Assert.Equal(target, path);
    }

    private static string Sha256Of(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
}
