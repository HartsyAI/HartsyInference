using HartsyInference.Core.Exceptions;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The residency planner's fit matrix on synthetic demands: resident, streamed and infeasible at their exact boundaries, the account a refusal
/// names, and the placements the host admits.</summary>
public sealed class ResidencyPlannerTests
{
    // The working set (KvCache 20, Activations 30) and the headroom (10) need 60 bytes; the mapped weights are 1100.
    private const long Needed = 60;
    private const long Mapped = 1100;

    private static ResidencyDemand Demand() => new()
    {
        DenseBytes = 100,
        ExpertBytes = 1000,
        EngramBytes = 50,
        HeadroomBytes = 10,
        WorkingSet = new Dictionary<ResidencyAccount, long> { [ResidencyAccount.KvCache] = 20, [ResidencyAccount.Activations] = 30 },
    };

    [Fact]
    public void EverythingWithinMemory_IsResident()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed + Mapped);

        Assert.Equal(MemoryFitVerdict.Resident, plan.Verdict);
        Assert.Empty(plan.RefusedByAccount);
    }

    [Fact]
    public void MappedWeightsOneByteShort_AreStreamed()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed + Mapped - 1);

        Assert.Equal(MemoryFitVerdict.Streamed, plan.Verdict);
        Assert.Empty(plan.RefusedByAccount);
    }

    [Fact]
    public void WorkingSetExactlyFitting_IsFeasible()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed);

        Assert.Equal(MemoryFitVerdict.Streamed, plan.Verdict);
        Assert.Empty(plan.RefusedByAccount);
    }

    [Fact]
    public void OneByteShort_IsInfeasible_AndNamesTheAccountThatIsRefused()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed - 1);

        Assert.Equal(MemoryFitVerdict.Infeasible, plan.Verdict);
        // Headroom and KvCache fit in 59 bytes; Activations (30) does not fit in the 29 that remain.
        Assert.Single(plan.RefusedByAccount);
        Assert.Equal(30L, plan.RefusedByAccount[ResidencyAccount.Activations]);
        Assert.Contains("Activations", plan.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void ReservedByAccount_HoldsTheWorkingSetAndTheHeadroom()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed + Mapped);

        Assert.Equal(20L, plan.ReservedByAccount[ResidencyAccount.KvCache]);
        Assert.Equal(30L, plan.ReservedByAccount[ResidencyAccount.Activations]);
        Assert.Equal(10L, plan.ReservedByAccount[ResidencyAccount.Headroom]);
        Assert.Equal(Needed, plan.WorkingSetBytes + plan.HeadroomBytes);
        Assert.Equal(Mapped, plan.MappedBytes);
    }

    [Fact]
    public void NegativeAvailableMemory_IsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ResidencyPlanner.Plan(Demand(), availableBytes: -1));
    }

    [Fact]
    public void HostPlacements_AreAdmitted()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed + Mapped);

        ResidencyPlanner.RequireAdmitted(plan);
        Assert.Contains(plan.Components, component => component.Component == ResidencyComponent.RoutedExperts && component.Mode == ResidencyMode.CpuExecute);
    }

    [Fact]
    public void ExpertsOnAGpuDevice_AreRefused()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed + Mapped);
        ResidencyPlan onGpu = plan with { Components = [new ComponentResidency(ResidencyComponent.RoutedExperts, ResidencyMode.Resident, "cuda:0", 1000)] };

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => ResidencyPlanner.RequireAdmitted(onGpu));
        Assert.Contains("cuda:0", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StreamedExpertsOnTheHost_AreRefused()
    {
        ResidencyPlan plan = ResidencyPlanner.Plan(Demand(), availableBytes: Needed + Mapped);
        ResidencyPlan streamed = plan with { Components = [new ComponentResidency(ResidencyComponent.RoutedExperts, ResidencyMode.Streamed, "cpu", 1000)] };

        Assert.Throws<HartsyInferenceException>(() => ResidencyPlanner.RequireAdmitted(streamed));
    }
}
