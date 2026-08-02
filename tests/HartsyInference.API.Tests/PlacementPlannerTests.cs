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
}
