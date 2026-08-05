using HartsyInference.Engine.Placement;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Split-math contract for the layer-split planner. Uses explicit ratios throughout so no CUDA probe
/// runs — these are Unit-tier and must pass on a GPU-less host.</summary>
public sealed class PlacementPlannerTests
{
    [Fact]
    public void ExplicitRatios_SplitProportionally_CoveringAllLayers()
    {
        IReadOnlyList<LlmStagePlan> plan = PlacementPlanner.LlmSplitPlan(
            ["cuda:0", "cuda:1"], [0.75f, 0.25f], layerCount: 32, perLayerBytes: 100 << 20);

        Assert.Equal(2, plan.Count);
        Assert.Equal(0, plan[0].StartLayer);
        Assert.Equal(plan[0].EndLayer, plan[1].StartLayer);
        Assert.Equal(32, plan[1].EndLayer);
        Assert.Equal(24, plan[0].EndLayer - plan[0].StartLayer);
        Assert.Equal(8, plan[1].EndLayer - plan[1].StartLayer);
    }

    [Fact]
    public void EveryStage_GetsAtLeastOneLayer_EvenAtExtremeRatios()
    {
        IReadOnlyList<LlmStagePlan> plan = PlacementPlanner.LlmSplitPlan(
            ["cuda:0", "cuda:1", "cuda:2"], [1f, 0.001f, 0.001f], layerCount: 10, perLayerBytes: 1);

        Assert.Equal(3, plan.Count);
        int total = 0;
        foreach (LlmStagePlan stage in plan)
        {
            int layers = stage.EndLayer - stage.StartLayer;
            Assert.True(layers >= 1, $"stage {stage.Device} got {layers} layers");
            total += layers;
        }
        Assert.Equal(10, total);
    }

    [Fact]
    public void ContiguousRanges_InDeviceOrder()
    {
        IReadOnlyList<LlmStagePlan> plan = PlacementPlanner.LlmSplitPlan(
            ["cuda:1", "cuda:0"], [0.5f, 0.5f], layerCount: 9, perLayerBytes: 1);

        Assert.Equal("cuda:1", plan[0].Device);
        Assert.Equal("cuda:0", plan[1].Device);
        Assert.Equal(0, plan[0].StartLayer);
        Assert.Equal(plan[0].EndLayer, plan[1].StartLayer);
        Assert.Equal(9, plan[1].EndLayer);
    }

    [Fact]
    public void FewerLayersThanStages_Throws()
    {
        Assert.Throws<ArgumentException>(() => PlacementPlanner.LlmSplitPlan(
            ["cuda:0", "cuda:1", "cuda:2"], [1f, 1f, 1f], layerCount: 2, perLayerBytes: 1));
    }

    [Fact]
    public void RatioCountMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => PlacementPlanner.LlmSplitPlan(
            ["cuda:0", "cuda:1"], [1f], layerCount: 8, perLayerBytes: 1));
    }

    private const long Reserve = 2L << 30;
    private const long MiB = 1L << 20;

    /// <summary>Chroma-shaped heterogeneity: 19 double blocks at 2× the 38 single blocks. Equal budgets must
    /// split at the byte midpoint (block 19), not the count midpoint (28/29) the count-proportional overload picks.</summary>
    [Fact]
    public void ByteWeightedDitSplit_HeterogeneousBlocks_SplitsByBytesNotCount()
    {
        long[] perBlock = new long[57];
        for (int i = 0; i < 19; i++) perBlock[i] = 200 * MiB;
        for (int i = 19; i < 57; i++) perBlock[i] = 100 * MiB;

        int split = PlacementPlanner.DitSplitPlan(Reserve + (1L << 30), Reserve + (1L << 30), perBlock, sharedWeightBytesA: 0);

        Assert.Equal(19, split);
    }

    [Fact]
    public void ByteWeightedDitSplit_SharedWeightsChargeStageA()
    {
        long[] perBlock = new long[10];
        for (int i = 0; i < 10; i++) perBlock[i] = 100 * MiB;

        int withoutShared = PlacementPlanner.DitSplitPlan(Reserve + (2L << 30), Reserve + (1L << 30), perBlock, sharedWeightBytesA: 0);
        int withShared = PlacementPlanner.DitSplitPlan(Reserve + (2L << 30), Reserve + (1L << 30), perBlock, sharedWeightBytesA: 1L << 30);

        Assert.True(withShared < withoutShared,
            $"charging shared bytes to A must shrink A's range ({withShared} !< {withoutShared})");
        Assert.Equal(5, withShared);
    }

    [Fact]
    public void ByteWeightedDitSplit_ExtremeBudgets_FloorAtOneBlockPerStage()
    {
        long[] perBlock = new long[8];
        for (int i = 0; i < 8; i++) perBlock[i] = 100 * MiB;

        Assert.Equal(1, PlacementPlanner.DitSplitPlan(Reserve + 1, Reserve + (10L << 30), perBlock, 0));
        Assert.Equal(7, PlacementPlanner.DitSplitPlan(Reserve + (10L << 30), Reserve + 1, perBlock, 0));
    }

    [Fact]
    public void ByteWeightedDitSplit_NoVramSignal_FallsBackToEvenByteSplit()
    {
        long[] perBlock = new long[10];
        for (int i = 0; i < 10; i++) perBlock[i] = 100 * MiB;

        Assert.Equal(5, PlacementPlanner.DitSplitPlan(0, 0, perBlock, 0));
    }

    [Fact]
    public void ByteWeightedDitSplit_FewerThanTwoBlocks_Throws()
    {
        Assert.Throws<ArgumentException>(() => PlacementPlanner.DitSplitPlan(Reserve, Reserve, new long[] { 100 * MiB }, 0));
    }
}
