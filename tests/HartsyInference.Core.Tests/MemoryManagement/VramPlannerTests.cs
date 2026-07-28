using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.MemoryManagement;

/// <summary>Decision-table tests for <see cref="VramPlanner"/>: the placement it returns is a pure function of
/// (policy, free bytes, weight bytes, residency, streamability), so a fake cache reporting a fixed budget covers it
/// with no GPU.</summary>
public sealed class VramPlannerTests
{
    /// <summary>Fake cache that reports a fixed device budget, so a test can state "the card has N bytes free" directly.</summary>
    /// <remarks>Mirrors the real <c>CudaStreamingWeightCache</c> contract: available = free − reserve, floored at zero.</remarks>
    private sealed class BudgetCache : IStreamingWeightCache
    {
        private readonly long _freeBytes;

        public BudgetCache(long freeBytes) => _freeBytes = freeBytes;

        public long LastReserveQueried { get; private set; } = -1;

        public long QueryAvailableWeightCacheBytes(long activationReserve)
        {
            LastReserveQueried = activationReserve;
            long available = _freeBytes - activationReserve;
            return available < 0 ? 0 : available;
        }

        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights) => StreamingUploadToken.Empty;

        public void AwaitWeights(StreamingUploadToken token) { }

        public void EvictAsync(IEnumerable<Tensor> weights) { }

        public void DrainAndReleasePool() { }
    }

    private const long Mb = 1024 * 1024;

    [Fact]
    public void FitsResident_ChoosesResident()
    {
        VramPlanner planner = new VramPlanner(new BudgetCache(8000 * Mb), "test", LowVramMode.Auto);
        Assert.Equal(PhasePlacement.Resident, planner.PlanPhase("denoise", weightBytes: 4000 * Mb, activationReserveBytes: 2000 * Mb, alreadyResident: false, canStream: true));
    }

    [Fact]
    public void DoesNotFitResident_ChoosesStreamed()
    {
        VramPlanner planner = new VramPlanner(new BudgetCache(8000 * Mb), "test", LowVramMode.Auto);
        Assert.Equal(PhasePlacement.Streamed, planner.PlanPhase("denoise", weightBytes: 12000 * Mb, activationReserveBytes: 2000 * Mb, alreadyResident: false, canStream: true));
    }

    /// <summary>The activation reserve is subtracted before the weights are considered — a phase whose weights alone
    /// would fit must still stream when the reserve pushes it over.</summary>
    [Fact]
    public void ActivationReserve_CountsAgainstTheWeightBudget()
    {
        BudgetCache cache = new BudgetCache(8000 * Mb);
        VramPlanner planner = new VramPlanner(cache, "test", LowVramMode.Auto);
        Assert.Equal(PhasePlacement.Streamed, planner.PlanPhase("denoise", weightBytes: 7000 * Mb, activationReserveBytes: 3000 * Mb, alreadyResident: false, canStream: true));
        Assert.Equal(3000 * Mb, cache.LastReserveQueried);
    }

    /// <summary>Regression guard for the oscillation described at <c>FluxPipeline.cs:489-495</c>: querying availability
    /// for weights that are themselves occupying the space reports "does not fit", flipping warm generations between
    /// resident and streamed. An already-resident phase must never reach the query.</summary>
    [Fact]
    public void AlreadyResident_SkipsTheAvailabilityQueryEntirely()
    {
        BudgetCache cache = new BudgetCache(0);
        VramPlanner planner = new VramPlanner(cache, "test", LowVramMode.Auto);
        Assert.Equal(PhasePlacement.Resident,
            planner.PlanPhase("denoise", weightBytes: 12000 * Mb, activationReserveBytes: 2000 * Mb, alreadyResident: true, canStream: true));
        Assert.Equal(-1, cache.LastReserveQueried);
    }

    /// <summary>The power-user escape hatch: never stream, never auto-evict, let an oversized model fail.</summary>
    [Fact]
    public void ForceOff_StaysResidentEvenWhenItCannotFit()
    {
        BudgetCache cache = new BudgetCache(0);
        VramPlanner planner = new VramPlanner(cache, "test", LowVramMode.ForceOff);
        Assert.Equal(PhasePlacement.Resident, planner.PlanPhase("denoise", weightBytes: 99000 * Mb, activationReserveBytes: 2000 * Mb, alreadyResident: false, canStream: true));
        Assert.Equal(-1, cache.LastReserveQueried);
        Assert.False(planner.CanStream);
    }

    [Fact]
    public void ForceOn_StreamsEvenWhenItWouldFit()
    {
        VramPlanner planner = new VramPlanner(new BudgetCache(99000 * Mb), "test", LowVramMode.ForceOn);
        Assert.Equal(PhasePlacement.Streamed, planner.PlanPhase("denoise", weightBytes: 1 * Mb, activationReserveBytes: 1 * Mb, alreadyResident: false, canStream: true));
    }

    /// <summary>CPU and Vulkan have no device weight cache; they must keep today's fully-resident behavior.</summary>
    [Fact]
    public void NoStreamingCache_AlwaysResident()
    {
        VramPlanner planner = new VramPlanner(cache: null, "test", LowVramMode.ForceOn);
        Assert.Equal(PhasePlacement.Resident, planner.PlanPhase("denoise", weightBytes: 99000 * Mb, activationReserveBytes: 2000 * Mb, alreadyResident: false, canStream: true));
        Assert.False(planner.CanStream);
    }

    /// <summary>A denoiser with no <see cref="IStreamingBlock"/> decomposition cannot stream however tight VRAM is.</summary>
    [Fact]
    public void NotStreamable_StaysResidentEvenWhenItDoesNotFit()
    {
        VramPlanner planner = new VramPlanner(new BudgetCache(0), "test", LowVramMode.Auto);
        Assert.Equal(PhasePlacement.Resident,
            planner.PlanPhase("denoise", weightBytes: 99000 * Mb, activationReserveBytes: 2000 * Mb, alreadyResident: false, canStream: false));
    }

    [Fact]
    public void ThrowInfeasible_NamesThePhaseAndSaysStreamingCannotHelp()
    {
        VramPlanner planner = new VramPlanner(new BudgetCache(512 * Mb), "test", LowVramMode.Auto);
        OutOfVramException ex = Assert.Throws<OutOfVramException>(() => planner.ThrowInfeasible("vae-decode", 9000 * Mb));
        Assert.Contains("vae-decode", ex.Message, StringComparison.Ordinal);
        Assert.Contains("streaming cannot", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(null, LowVramMode.Auto)]
    [InlineData("", LowVramMode.Auto)]
    [InlineData("auto", LowVramMode.Auto)]
    [InlineData("nonsense", LowVramMode.Auto)]
    [InlineData("1", LowVramMode.ForceOn)]
    [InlineData("on", LowVramMode.ForceOn)]
    [InlineData("TRUE", LowVramMode.ForceOn)]
    [InlineData("0", LowVramMode.ForceOff)]
    [InlineData("off", LowVramMode.ForceOff)]
    [InlineData("False", LowVramMode.ForceOff)]
    public void Policy_ParsesEveryDocumentedSpelling(string? value, LowVramMode expected)
    {
        string? previous = Environment.GetEnvironmentVariable(LowVramPolicy.EnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(LowVramPolicy.EnvironmentVariable, value);
            LowVramPolicy.ResetCacheForTests();
            Assert.Equal(expected, LowVramPolicy.Resolve());
        }
        finally
        {
            Environment.SetEnvironmentVariable(LowVramPolicy.EnvironmentVariable, previous);
            LowVramPolicy.ResetCacheForTests();
        }
    }
}
