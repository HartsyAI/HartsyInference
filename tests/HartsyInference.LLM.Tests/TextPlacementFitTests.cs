using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>How a placement reads as a memory fit. A host routes on the verdict, so offload reporting as resident would send a
/// request to a card that runs part of the model on the CPU, and an infeasible plan reporting as anything else would load into
/// an out-of-memory error.</summary>
public sealed class TextPlacementFitTests
{
    private static TextPlacement Plan(TextPlacementMode mode, bool feasible) =>
        new(mode, feasible, ["cuda"], 0, 20L << 30, 22L << 30, "reason");

    [Theory]
    [InlineData(TextPlacementMode.Gpu, true, MemoryFitVerdict.Resident)]
    [InlineData(TextPlacementMode.Split, true, MemoryFitVerdict.Resident)]
    [InlineData(TextPlacementMode.Offload, true, MemoryFitVerdict.Streamed)]
    [InlineData(TextPlacementMode.Gpu, false, MemoryFitVerdict.Infeasible)]
    [InlineData(TextPlacementMode.Split, false, MemoryFitVerdict.Infeasible)]
    [InlineData(TextPlacementMode.Offload, false, MemoryFitVerdict.Infeasible)]
    public void PlacementMapsToTheVerdictAHostRoutesOn(TextPlacementMode mode, bool feasible, MemoryFitVerdict expected)
    {
        MemoryFit fit = MemoryEstimationService.FitFromPlacement(Plan(mode, feasible), VramTier.Auto);
        Assert.Equal(expected, fit.Verdict);
        Assert.Equal(22L << 30, fit.CapacityBytes);
        Assert.Equal("reason", fit.Reason);
    }
}
