using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Features;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Request-supplied model names resolve under the models root. A relative name that climbs out of it with a
/// <c>..</c> segment is refused, while an absolute path that exists is still returned, because the CLI and SwarmUI pass those.
/// Points the models root at a temporary fixture, so nothing here reads a real file.</summary>
[Collection("ModelsRootKnob")]
public sealed class ModelFileLocatorTests : IDisposable
{
    private readonly string _parent = Path.Combine(Path.GetTempPath(), "hartsy-locator-tests-" + Path.GetRandomFileName());
    private readonly string _outside;
    private readonly string _adapter;

    public ModelFileLocatorTests()
    {
        // Load the settings file first: its first load would overwrite the override set below.
        _ = EngineKnobs.ModelsRoot.Value;
        string root = Path.Combine(_parent, "Models");
        Directory.CreateDirectory(Path.Combine(root, "Lora"));
        _outside = Path.Combine(_parent, "outside.safetensors");
        _adapter = Path.Combine(root, "Lora", "adapter.safetensors");
        File.WriteAllText(_outside, "outside");
        File.WriteAllText(_adapter, "adapter");
        KnobStore.Set(EngineKnobs.ModelsRoot, root);
    }

    public void Dispose()
    {
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        if (Directory.Exists(_parent))
            Directory.Delete(_parent, recursive: true);
    }

    [Fact]
    public void RelativeNameInsideTheRootResolves()
    {
        string? found = ModelFileLocator.Find("adapter.safetensors", "Lora");
        Assert.NotNull(found);
        Assert.Equal(Path.GetFullPath(_adapter), Path.GetFullPath(found));
    }

    [Theory]
    [InlineData("../outside.safetensors")]
    [InlineData("..\\outside.safetensors")]
    [InlineData("Lora/../../outside.safetensors")]
    public void RelativeNameThatClimbsOutOfTheRootIsRefused(string name)
    {
        Assert.Null(ModelFileLocator.Find(name, "Lora"));
    }

    [Fact]
    public void AbsolutePathThatExistsIsStillReturned()
    {
        string? found = ModelFileLocator.Find(_outside);
        Assert.NotNull(found);
        Assert.Equal(Path.GetFullPath(_outside), Path.GetFullPath(found));
    }
}
