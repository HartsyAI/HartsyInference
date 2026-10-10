using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>How a Text model directory resolves: a single .gguf as before, a Hugging Face checkpoint as itself, and both
/// together as an explicit ambiguity error.</summary>
[Collection("ModelsRootKnob")]
public sealed class DeepSeekV41TextResolutionTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("dsv41-resolver-").FullName;

    // The root to put back afterwards. (A Set now loads the settings file itself, so no read is needed before it.)
    private readonly string? _previousRoot = EngineKnobs.ModelsRoot.Value;

    public DeepSeekV41TextResolutionTests() => KnobStore.Set(EngineKnobs.ModelsRoot, _root);

    public void Dispose()
    {
        if (_previousRoot is null)
            KnobStore.Clear(EngineKnobs.ModelsRoot);
        else
            KnobStore.Set(EngineKnobs.ModelsRoot, _previousRoot);
        Directory.Delete(_root, recursive: true);
    }

    private string ModelDirectory(string id)
    {
        string directory = Path.Combine(_root, "LLM", id);
        Directory.CreateDirectory(directory);
        return directory;
    }

    [Fact]
    public void SingleGguf_ResolvesToTheGgufFileAsBefore()
    {
        string directory = ModelDirectory("some-model");
        string gguf = Path.Combine(directory, "model-Q4.gguf");
        File.WriteAllBytes(gguf, new byte[4]);

        Assert.Equal(gguf, ModelResolver.Resolve("some-model", null, Modality.Text).LocalPath);
        Assert.Equal(gguf, ModelResolver.ResolveTextDirectory(directory));
    }

    [Fact]
    public void TwoGgufs_AreStillAmbiguousAndResolveToNothing()
    {
        string directory = ModelDirectory("some-model");
        File.WriteAllBytes(Path.Combine(directory, "a.gguf"), new byte[4]);
        File.WriteAllBytes(Path.Combine(directory, "b.gguf"), new byte[4]);

        Assert.Null(ModelResolver.Resolve("some-model", null, Modality.Text).LocalPath);
    }

    [Fact]
    public void CatalogIdWithAHfCheckpointDirectory_ResolvesToTheDirectory()
    {
        string directory = ModelDirectory(DeepSeekV41Catalog.Id);
        TinyDeepSeekV41Checkpoint.Write(directory);

        ModelSpec spec = ModelResolver.Resolve(DeepSeekV41Catalog.Id, null, Modality.Text);

        Assert.Equal(directory, spec.LocalPath);
        Assert.Same(ModelCatalog.Find(DeepSeekV41Catalog.Id), spec.Catalog);
    }

    [Fact]
    public void GgufBesideASafetensorsIndex_IsRefusedAsAmbiguous()
    {
        string directory = ModelDirectory(DeepSeekV41Catalog.Id);
        TinyDeepSeekV41Checkpoint.Write(directory);
        File.WriteAllBytes(Path.Combine(directory, "DeepSeek-V4.1-Flash-Q2.gguf"), new byte[4]);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => ModelResolver.Resolve(DeepSeekV41Catalog.Id, null, Modality.Text));

        Assert.Contains("model.safetensors.index.json", error.Message);
        Assert.Contains(".gguf", error.Message);
        Assert.Contains("--model-path", error.Message);
    }

}
