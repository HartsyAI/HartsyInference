using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Resolves the <c>qwen3</c> catalog id against the real models root (<c>HARTSYINFERENCE_MODELS_DIR</c>), read
/// only. A SwarmUI-managed root keeps the file under <c>llm/qwen3/</c> beside a second GGUF, which is the layout the
/// engine could not resolve on a case-sensitive filesystem. Skips when the catalog's file is not in the store.</summary>
[Collection("ModelsRootKnob")]
[Trait("Category", "Integration")]
public sealed class Qwen3ModelStoreResolutionTests : IDisposable
{
    private readonly ITestOutputHelper _output;

    // The root to put back afterwards. (A Set now loads the settings file itself, so no read is needed before it.)
    private readonly string? _previousRoot = EngineKnobs.ModelsRoot.Value;

    public Qwen3ModelStoreResolutionTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        if (_previousRoot is null)
            KnobStore.Clear(EngineKnobs.ModelsRoot);
        else
            KnobStore.Set(EngineKnobs.ModelsRoot, _previousRoot);
    }

    [Fact]
    public void Qwen3_ResolvesToTheCatalogFileInTheRealStore()
    {
        string fileName = ModelCatalog.Find("qwen3")!.Assets[0].FileName;
        string lower = Path.Combine(TestPaths.ModelsDir, "llm", "qwen3", fileName);
        string upper = Path.Combine(TestPaths.ModelsDir, "LLM", "qwen3", fileName);
        string expected = File.Exists(upper) ? upper : lower;
        if (!RealWeightGate.Require(_output.WriteLine, expected))
            return;
        KnobStore.Set(EngineKnobs.ModelsRoot, TestPaths.ModelsDir);

        ModelSpec spec = ModelResolver.Resolve("qwen3", modelPathArg: null, Modality.Text);

        _output.WriteLine($"qwen3 -> {spec.LocalPath}");
        Assert.Equal(Path.GetFullPath(expected), spec.LocalPath);
    }
}
