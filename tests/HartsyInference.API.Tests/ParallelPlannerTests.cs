using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Placement;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Pins <see cref="ParallelPlanner"/>'s measured-fact rules on synthetic topologies (no GPU): fit
/// beats latency, latency strategies require balanced NVLink-class fabric (the dev pair's measured losses are
/// the justification), CFG-parallel is video-only with strict secondary fit, and every plan passes
/// <see cref="PlacementPlanner.ValidatePlacement"/>.</summary>
public sealed class ParallelPlannerTests
{
    private static GpuTopologyInfo Gpu(int ordinal, int sm, long freeGb, int ccMajor = 8, int ccMinor = 9) =>
        new(ordinal, $"gpu{ordinal}", freeGb << 30, freeGb << 30, ccMajor, ccMinor, sm);

    private static GpuLinkInfo Link(int from, int to, bool nvlink) =>
        new(from, to, PeerAccessSupported: nvlink, PerformanceRank: nvlink ? 0 : -1, NativeAtomics: nvlink, LikelyNvLink: nvlink);

    private static readonly GpuTopologyInfo[] DevPair = [Gpu(0, 128, 22), Gpu(1, 28, 11, ccMinor: 6)];
    private static readonly GpuLinkInfo[] NoP2P = [Link(0, 1, false), Link(1, 0, false)];
    private static readonly GpuTopologyInfo[] NvlinkPair = [Gpu(0, 132, 70), Gpu(1, 132, 70)];
    private static readonly GpuLinkInfo[] Nvlink = [Link(0, 1, true), Link(1, 0, true)];

    private static ParallelPlan Plan(Modality modality, long modelGb, GpuTopologyInfo[] gpus, GpuLinkInfo[] links) =>
        ParallelPlanner.Suggest(new ParallelPlanRequest
        {
            Modality = modality,
            ModelBytes = modelGb << 30,
            Gpus = gpus,
            Links = links,
        });

    [Fact]
    public void SingleGpu_AlwaysSingle()
    {
        ParallelPlan plan = Plan(Modality.Video, 30, [Gpu(0, 128, 22)], []);
        Assert.True(plan.Placement.IsSingle, plan.Reason);
    }

    [Fact]
    public void Image_ModelExceedsPrimary_PicksDitSharding_FastestFirst()
    {
        ParallelPlan plan = Plan(Modality.Image, 30, DevPair, NoP2P);
        Assert.True(plan.Placement.EnableDitSharding, plan.Reason);
        Assert.Equal(["cuda:0", "cuda:1"], plan.Placement.ShardDevices);
        PlacementPlanner.ValidatePlacement(plan.Placement);
    }

    [Fact]
    public void Text_ModelExceedsPrimary_PicksLayerSplit()
    {
        ParallelPlan plan = Plan(Modality.Text, 30, DevPair, NoP2P);
        Assert.False(plan.Placement.EnableDitSharding);
        Assert.Equal(2, plan.Placement.ShardDevices.Count);
        Assert.Equal(1, plan.Placement.TensorParallelDegree);
        PlacementPlanner.ValidatePlacement(plan.Placement);
    }

    [Fact]
    public void Text_Fits_OnUnbalancedNoP2P_StaysSingle()
    {
        // The measured verdict: TP all-reduces are link-latency-bound; PCIe + unbalanced pair loses.
        ParallelPlan plan = Plan(Modality.Text, 1, DevPair, NoP2P);
        Assert.True(plan.Placement.IsSingle, plan.Reason);
    }

    [Fact]
    public void Text_Fits_OnBalancedNvlink_PicksTensorParallel()
    {
        ParallelPlan plan = Plan(Modality.Text, 20, NvlinkPair, Nvlink);
        Assert.Equal(2, plan.Placement.TensorParallelDegree);
        Assert.Equal(2, plan.Placement.ShardDevices.Count);
        PlacementPlanner.ValidatePlacement(plan.Placement);
    }

    [Fact]
    public void Video_Fits_OnBalancedNvlink_PicksContextParallel()
    {
        ParallelPlan plan = Plan(Modality.Video, 10, NvlinkPair, Nvlink);
        Assert.Equal(2, plan.Placement.ContextParallelDevices.Count);
        PlacementPlanner.ValidatePlacement(plan.Placement);
    }

    [Fact]
    public void Video_FitsBothCards_OnNoP2P_PicksCfgParallel()
    {
        // 5 GB model: fits the 3060 with the 1.3x margin → the measured Wan-class per-step CFG win applies.
        ParallelPlan plan = Plan(Modality.Video, 5, DevPair, NoP2P);
        Assert.Equal("cuda:1", plan.Placement.CfgParallelDevice);
        PlacementPlanner.ValidatePlacement(plan.Placement);
    }

    [Fact]
    public void Video_ReplicaTightOnSecondary_StaysSingle()
    {
        // 10 GB model on an 11 GB secondary: 1.3x margin fails — the measured SDXL 2.6x-slower trap.
        ParallelPlan plan = Plan(Modality.Video, 10, DevPair, NoP2P);
        Assert.True(plan.Placement.IsSingle, plan.Reason);
    }

    [Fact]
    public void Image_Fits_OnNoP2P_StaysSingle_NoCfgAuto()
    {
        // Image CFG-parallel is deliberately never auto-picked (SDXL measured 2.6x slower).
        ParallelPlan plan = Plan(Modality.Image, 2, DevPair, NoP2P);
        Assert.True(plan.Placement.IsSingle, plan.Reason);
    }

    [Fact]
    public void UnknownModelSize_AssumesFits_NeverPicksFitStrategies()
    {
        ParallelPlan plan = Plan(Modality.Image, 0, DevPair, NoP2P);
        Assert.False(plan.Placement.EnableDitSharding, plan.Reason);
    }

    [Fact]
    public void OtherModalities_Single_WithPointerReason()
    {
        ParallelPlan plan = Plan(Modality.Music, 4, DevPair, NoP2P);
        Assert.True(plan.Placement.IsSingle);
        Assert.Contains("ShardDevices", plan.Reason);
    }
}
