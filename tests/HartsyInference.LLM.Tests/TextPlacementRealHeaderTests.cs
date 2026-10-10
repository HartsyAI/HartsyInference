using HartsyInference.Engine.Placement;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>The demand the planner reads from a real MoE header, against figures taken from the same file with an independent
/// parser: Qwen3-30B-A3B Q4_K_M holds 17.55 GB of routed experts and about 1.0 GB of other weights.</summary>
[Trait("Category", "Integration")]
public sealed class TextPlacementRealHeaderTests(ITestOutputHelper output)
{
    [Fact]
    public void Qwen3MoeHeader_SplitsExpertsFromDenseWeights()
    {
        string path = Path.Combine(HartsyInference.Engine.RepoPaths.ModelsRoot(),
            "llm/moe-parity/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf");
        if (!RealWeightGate.Require(output.WriteLine, path)) return;
        TextPlacementDemand demand = TextPlacementDemandReader.FromGguf(path, 8192, includeRedundantSplits: false, kvF16: false);
        output.WriteLine($"dense {demand.DenseBytes / 1e9:F3} GB, experts {demand.ExpertBytes / 1e9:F3} GB, layers {demand.LayerCount}, "
            + $"KV/token {demand.KvBytesPerToken} B");
        Assert.InRange(demand.ExpertBytes / 1e9, 17.50, 17.60);
        Assert.InRange(demand.DenseBytes / 1e9, 0.3, 1.1);   // the untied token embedding stays on the host
        Assert.Equal(48, demand.LayerCount);
        Assert.Equal(48L * 4 * (128 + 128) * 4, demand.KvBytesPerToken);
    }
}
