using HartsyInference.ModelAssets.Gguf;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

[Trait("Category", "Integration")]
public sealed class GgufMuseSparkSplitTests
{
    private readonly ITestOutputHelper _output;

    public GgufMuseSparkSplitTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Load_RealSplitSet_IndexesEveryTensorAndResolvesTheMuseSparkMapper()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.Llm.MuseSparkOpenQ4KS)) return;

        using GgufLoader loader = new GgufLoader();
        loader.Load(TestPaths.Llm.MuseSparkOpenQ4KS);

        Assert.Equal(1429, loader.Descriptors.Count);
        Assert.Equal("muse-spark", loader.Metadata.GetString("general.architecture"));
        Assert.Equal("muse-spark", GgufModelLoader.ResolveMapping(loader).Architecture);
    }
}
