using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Preserves planner priority, rejection and execution boundaries while declarations move between files.</summary>
public sealed class AttentionPlannerContractTests
{
    [Fact]
    public void PlanningExplainsRejectionsWithoutLaunching()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        AttentionKernelRow[] rows =
        [
            Row(Capability("prefill") with { MinQueryTokens = 2 }, static (in AttentionCall _) => true),
            Row(Capability("unavailable"), static (in AttentionCall _) => false),
            Row(Capability("ready"), static (in AttentionCall _) => true),
        ];
        AttentionPlan plan = AttentionPlanner.Plan(rows, in call);
        Assert.Same(rows[2], plan.Row);
        Assert.Contains("prefill", plan.Declined);
        Assert.Contains("QueryRows", plan.Declined);
        Assert.Contains("unavailable", plan.Declined);
        Assert.Contains("ready", plan.Describe());
        Assert.All(operands.Output.AsReadOnlySpan<float>().ToArray(), value => Assert.Equal(7, value));
    }

    [Fact]
    public void ARefusedFormatCannotReachItsAdmissionOrLaunch()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call with { Numerics = new() { Refused = AttentionFormats.Tf32 } };
        AttentionKernelRow[] rows =
        [
            Row(Capability("tf32") with { UsesQueryKey = AttentionFormats.Tf32 },
                static (in AttentionCall _) => throw new InvalidOperationException("Excluded admission ran.")),
            new DelegateAttentionRow(Capability("f32"), null, static (in AttentionCall _) => true,
                static (in AttentionCall _) => true),
        ];
        Assert.Same(rows[1], AttentionPlanner.Run(rows, in call));
    }

    [Fact]
    public void F32NarrowingNeedsIndependentPermissionsForBothMultiplies()
    {
        using Operands operands = new();
        AttentionKernelCapability capability = Capability("narrow") with { NarrowsF32To = AttentionFormats.F16 };
        AttentionCall exact = operands.Call;
        AttentionCall queryKeyOnly = exact with { Numerics = new() { QueryKey = AttentionFormats.F16 } };
        AttentionCall both = exact with
        {
            Numerics = new() { QueryKey = AttentionFormats.F16, ProbValue = AttentionFormats.F16 },
        };

        Assert.Equal(AttentionExclusion.Numerics, capability.Excludes(in exact));
        Assert.Equal(AttentionExclusion.Numerics, capability.Excludes(in queryKeyOnly));
        Assert.Equal(AttentionExclusion.None, capability.Excludes(in both));
    }

    [Fact]
    public void RecoverableDeclinePreservesPreferenceOrder()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        List<string> executed = [];
        AttentionKernelRow[] rows =
        [
            new DelegateAttentionRow(Capability("unavailable"), null, static (in AttentionCall _) => false,
                (in AttentionCall _) => { executed.Add("unavailable"); return true; }),
            new DelegateAttentionRow(Capability("decline"), null, static (in AttentionCall _) => true,
                (in AttentionCall _) => { executed.Add("decline"); return false; }),
            new DelegateAttentionRow(Capability("ready"), null, static (in AttentionCall _) => true,
                (in AttentionCall _) => { executed.Add("ready"); return true; }),
            Row(Capability("unused"), static (in AttentionCall _) => throw new InvalidOperationException("Later admission ran.")),
        ];
        Assert.Same(rows[2], AttentionPlanner.Run(rows, in call));
        Assert.Equal(new[] { "decline", "ready" }, executed);
    }

    [Fact]
    public void AThrownExecutionFailureDoesNotRetryAnotherRow()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        InvalidOperationException failure = new("Launch failed after work began.");
        AttentionKernelRow[] rows =
        [
            new DelegateAttentionRow(Capability("failed"), null, static (in AttentionCall _) => true,
                (in AttentionCall _) => throw failure),
            Row(Capability("unused"), static (in AttentionCall _) => throw new InvalidOperationException("Retried.")),
        ];
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() => AttentionPlanner.Run(rows, in call)));
    }

    [Fact]
    public void UnsupportedCallsLeaveOutputUntouched()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        AttentionKernelRow[] rows =
        [
            Row(Capability("prefill") with { MinQueryTokens = 2 },
                static (in AttentionCall _) => throw new InvalidOperationException("Excluded admission ran.")),
        ];
        Assert.Throws<AttentionUnsupportedException>(() => AttentionPlanner.Run(rows, in call));
        foreach (float value in operands.Output.AsReadOnlySpan<float>()) Assert.Equal(7, value);
    }

    [Fact]
    public void SuccessfulExecutionHasNoPerCallAllocationGrowth()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        AttentionKernelRow[] rows =
        [
            new DelegateAttentionRow(Capability("ready"), null, static (in AttentionCall _) => true,
                static (in AttentionCall _) => true),
        ];
        for (int i = 0; i < 100; i++) AttentionPlanner.Run(rows, in call);
        long beforeSmall = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) AttentionPlanner.Run(rows, in call);
        long smallBytes = GC.GetAllocatedBytesForCurrentThread() - beforeSmall;
        long beforeLarge = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10000; i++) AttentionPlanner.Run(rows, in call);
        long largeBytes = GC.GetAllocatedBytesForCurrentThread() - beforeLarge;
        // A fixed allowance absorbs instrumentation noise, while a single per-call allocation grows far beyond it.
        Assert.True(largeBytes <= smallBytes + 1024, $"Allocation grew from {smallBytes} to {largeBytes} bytes.");
    }

    private static AttentionKernelCapability Capability(string id) => new()
    {
        Id = id, Entry = AttentionRoute.Dense, QueryTypes = AttentionDTypes.F32,
        KvTypes = AttentionDTypes.F32, Layouts = AttentionLayouts.HeadMajor,
    };

    private static AttentionKernelRow Row(AttentionKernelCapability capability, AttentionRowFunc admits) =>
        new DelegateAttentionRow(capability, null, admits,
            static (in AttentionCall _) => throw new InvalidOperationException("Planning launched a kernel."));

    private sealed class Operands : IDisposable
    {
        private readonly Tensor _query = new(new TensorShape(1, 1, 1, 2), DType.F32);
        private readonly Tensor _key = new(new TensorShape(1, 1, 1, 2), DType.F32);
        private readonly Tensor _value = new(new TensorShape(1, 1, 1, 2), DType.F32);
        public Tensor Output { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);

        public Operands() => Output.AsSpan<float>().Fill(7);

        public AttentionCall Call => AttentionCall.FromScaledDotProductAttention(Output, _query, _key, _value, null, 1, false);

        public void Dispose()
        {
            Output.Dispose();
            _value.Dispose();
            _key.Dispose();
            _query.Dispose();
        }
    }
}
