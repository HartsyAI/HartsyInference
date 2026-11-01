using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>A catalog asset lands in, and is found under, the folder its model type is set to in the settings.</summary>
public sealed class ModelFolderSettingsTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"hartsy-folders-{Guid.NewGuid():N}");

    public ModelFolderSettingsTests()
    {
        Directory.CreateDirectory(_root);
        KnobStore.Set(EngineKnobs.ModelsRoot, _root);
    }

    public void Dispose()
    {
        KnobStore.ResetOverrides();
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void AssetFoundInTheFolderSetInSettings()
    {
        KnobStore.Set(EngineKnobs.ModelFolderLlm, "my-llms");
        string dir = Path.Combine(_root, "my-llms", "demo");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "model.gguf"), "x");

        Assert.Equal(Path.Combine(dir, "model.gguf"), ModelDownloader.TargetPath(Asset("LLM/demo")));
    }

    [Fact]
    public void AssetFoundInAFolderSpelledInAnotherCase()
    {
        string dir = Path.Combine(_root, "LLM", "demo");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "model.gguf"), "x");

        Assert.Equal(Path.Combine(dir, "model.gguf"), ModelDownloader.TargetPath(Asset("LLM/demo")));
    }

    private static ModelAsset Asset(string targetSubdir) => new()
    {
        Repo = "example/demo",
        RepoPath = "model.gguf",
        TargetSubdir = targetSubdir,
        Role = "transformer",
    };
}
