using HartsyInference.Core.Backends;
using HartsyInference.Engine.Placement;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Gating contract for <see cref="PlacementConfig.TensorParallelDegree"/> in
/// <see cref="PlacementPlanner.ValidatePlacement"/>: TP claims <see cref="PlacementConfig.ShardDevices"/> as
/// its rank list (count must equal the degree, layer split off) and composes with nothing else — every invalid
/// combination must fail at configuration time, because at generation time a misread config silently becomes a
/// layer split or a single-device run that still produces plausible text. Unit tier (pure validation).</summary>
public sealed class TensorParallelPlacementValidationTests
{
    [Fact]
    public void ValidatePlacement_AcceptsTpWithMatchingRankDevices()
    {
        PlacementPlanner.ValidatePlacement(new PlacementConfig
        {
            TensorParallelDegree = 2,
            ShardDevices = ["cuda:0", "cuda:1"],
        });
    }

    [Fact]
    public void ValidatePlacement_AcceptsDefaultDegreeOne()
    {
        PlacementPlanner.ValidatePlacement(PlacementConfig.Single);
        PlacementPlanner.ValidatePlacement(new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"] });
    }

    [Fact]
    public void ValidatePlacement_RejectsDegreeBelowOne()
    {
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(
            new PlacementConfig { TensorParallelDegree = 0 }));
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(
            new PlacementConfig { TensorParallelDegree = -1 }));
    }

    [Fact]
    public void ValidatePlacement_RejectsRankDeviceCountMismatch()
    {
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(
            new PlacementConfig { TensorParallelDegree = 2 }));
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(
            new PlacementConfig { TensorParallelDegree = 2, ShardDevices = ["cuda:0"] }));
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(
            new PlacementConfig { TensorParallelDegree = 2, ShardDevices = ["cuda:0", "cuda:1", "cuda:2"] }));
    }

    [Fact]
    public void ValidatePlacement_RejectsShardRatiosUnderTp()
    {
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(new PlacementConfig
        {
            TensorParallelDegree = 2,
            ShardDevices = ["cuda:0", "cuda:1"],
            ShardRatios = [0.5f, 0.5f],
        }));
    }

    [Fact]
    public void ValidatePlacement_RejectsCompositionWithOtherMultiGpuModes()
    {
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(new PlacementConfig
        {
            TensorParallelDegree = 2,
            ShardDevices = ["cuda:0", "cuda:1"],
            EnableDitSharding = true,
        }));
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(new PlacementConfig
        {
            TensorParallelDegree = 2,
            ShardDevices = ["cuda:0", "cuda:1"],
            CfgParallelDevice = "cuda:1",
        }));
        Assert.Throws<ArgumentException>(() => PlacementPlanner.ValidatePlacement(new PlacementConfig
        {
            TensorParallelDegree = 2,
            ShardDevices = ["cuda:0", "cuda:1"],
            ContextParallelDevices = ["cuda:0", "cuda:1"],
        }));
    }
}
