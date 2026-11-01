using HartsyInference.Core.Configuration;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>The model-folder map: a catalog's logical folder names become the names on disk, and a settings override
/// replaces the default.</summary>
public sealed class ModelFoldersTests : IDisposable
{
    public void Dispose() => KnobStore.ResetOverrides();

    [Theory]
    [InlineData("LLM/nanbeige43", "llm/nanbeige43")]
    [InlineData("Audio/AuK", "audio/AuK")]
    [InlineData("Stable-Diffusion/Lance/Lance_3B", "Stable-Diffusion/Lance/Lance_3B")]
    [InlineData("VAE", "VAE")]
    [InlineData("text_encoders", "text_encoders")]
    [InlineData("Embedding/qwen3-embedding", "Embedding/qwen3-embedding")]
    [InlineData("Unknown/x", "Unknown/x")]
    public void MapSubdir_UsesTheDefaultNameOnDisk(string catalogPath, string expected) =>
        Assert.Equal(expected, ModelFolders.MapSubdir(catalogPath));

    [Fact]
    public void MapSubdir_UsesTheSettingsOverride()
    {
        KnobStore.Set(EngineKnobs.ModelFolderLlm, "my-llms");

        Assert.Equal("my-llms/nanbeige43", ModelFolders.MapSubdir("LLM/nanbeige43"));
        Assert.Equal("my-llms", ModelFolders.MapName("llm"));
    }

    [Fact]
    public void EmptyOverrideFallsBackToTheDefault()
    {
        KnobStore.Set(EngineKnobs.ModelFolderAudio, "");

        Assert.Equal("audio", ModelFolders.Physical(ModelFolders.Audio));
    }

    [Theory]
    [InlineData("audio", "audio")]
    [InlineData("Audio", "audio")]
    [InlineData("Embedding", "Embedding")]
    [InlineData("Other", "Other")]
    public void MapName_IgnoresCaseAndKeepsUnknownNames(string name, string expected) =>
        Assert.Equal(expected, ModelFolders.MapName(name));

    [Fact]
    public void Under_JoinsTheFolderOnDiskWithTheRest() =>
        Assert.Equal(Path.Combine("root", "audio", "wake"), ModelFolders.Under("root", ModelFolders.Audio, "wake"));
}
