using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The planner over a fake row table: order, declared exclusions, runtime admission, recoverable declines, and the
/// unsupported error that ends a chain no row could serve.</summary>
public sealed class AttentionPlannerTests
{
    private static AttentionKernelCapability Cap(string id, AttentionDTypes q = AttentionDTypes.F32 | AttentionDTypes.F16) => new()
    {
        Id = id, Entry = AttentionRoute.Dense, QueryTypes = q, KvTypes = q, Layouts = AttentionLayouts.HeadMajor,
        Masks = AttentionMaskSupport.HeadSharedBias,
    };

    private static AttentionCall Call(DType dt, int d = 64, Tensor? bias = null, int hq = 2, int hkv = 2)
    {
        Tensor q = new(new TensorShape(1, hq, 4, d), dt), k = new(new TensorShape(1, hkv, 4, d), dt);
        Tensor v = new(new TensorShape(1, hkv, 4, d), dt), o = new(new TensorShape(1, hq, 4, d), dt);
        return new()
        {
            Query = AttentionOperand.HeadMajor(q), Key = AttentionOperand.HeadMajor(k), Value = AttentionOperand.HeadMajor(v),
            Output = AttentionOperand.HeadMajor(o), Scale = 0.125f, Mask = AttentionMask.Additive(bias), Route = AttentionRoute.Dense,
        };
    }

    private static DelegateAttentionRow Row(string id, bool admits, bool runs, List<string> log, AttentionKernelCapability? cap = null) =>
        new(cap ?? Cap(id), id, (in AttentionCall _) => admits, (in AttentionCall _) => { log.Add(id); return runs; });

    [Fact]
    public void TheFirstRowThatCoversAndAdmits_Runs()
    {
        List<string> log = [];
        AttentionKernelRow[] rows = [Row("a", admits: false, runs: true, log), Row("b", true, true, log), Row("c", true, true, log)];
        AttentionKernelRow ran = AttentionPlanner.Run(rows, Call(DType.F32));
        Assert.Equal("b", ran.Capability.Id);
        Assert.Equal(["b"], log);
    }

    [Fact]
    public void ARecoverableDecline_FallsToTheNextRow()
    {
        List<string> log = [];
        AttentionKernelRow[] rows = [Row("a", true, runs: false, log), Row("b", true, true, log)];
        Assert.Equal("b", AttentionPlanner.Run(rows, Call(DType.F32)).Capability.Id);
        Assert.Equal(["a", "b"], log);
    }

    [Fact]
    public void ADeclaredLimit_ExcludesTheRowBeforeItsAdmissionRuns()
    {
        List<string> log = [];
        bool asked = false;
        DelegateAttentionRow f32Only = new(Cap("f32", AttentionDTypes.F32), "f32", (in AttentionCall _) => asked = true,
            (in AttentionCall _) => { log.Add("f32"); return true; });
        AttentionKernelRow[] rows = [f32Only, Row("any", true, true, log)];
        Assert.Equal("any", AttentionPlanner.Run(rows, Call(DType.F16)).Capability.Id);
        Assert.False(asked);
        Assert.Equal(AttentionExclusion.QueryType, f32Only.Capability.Excludes(Call(DType.F16)));
    }

    [Fact]
    public void Exclusions_NameTheLimitAndAllocateNothing()
    {
        AttentionKernelCapability c = Cap("x") with { MinHeadDim = 64, MaxHeadDim = 128 };
        Assert.Equal(AttentionExclusion.HeadDim, c.Excludes(Call(DType.F32, d: 40)));
        Assert.Equal(AttentionExclusion.GroupedKv, c.Excludes(Call(DType.F32, hq: 4, hkv: 2)));
        using Tensor perHead = new(new TensorShape(1, 2, 4, 4), DType.F32);
        Assert.Equal(AttentionExclusion.PerHeadBias, c.Excludes(Call(DType.F32, bias: perHead)));
        AttentionCall call = Call(DType.F32);
        Assert.Equal(AttentionExclusion.Entry, c.Excludes(call with { Route = AttentionRoute.Flash }));
        Assert.Equal(AttentionExclusion.None, c.Excludes(call));
        AttentionCall excluded = Call(DType.F32, d: 40);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) _ = c.Excludes(excluded);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void NoRowServing_IsAnUnsupportedErrorNamingTheDeclines()
    {
        List<string> log = [];
        AttentionKernelRow[] rows = [Row("a", admits: false, runs: true, log), Row("b", true, runs: false, log)];
        AttentionUnsupportedException ex = Assert.Throws<AttentionUnsupportedException>(() => AttentionPlanner.Run(rows, Call(DType.F32)));
        Assert.Contains("a (not admitted), b (declined)", ex.Message);
    }

    [Fact]
    public void Plan_NamesTheRowRunWouldTryFirst_AndTheRowsAheadOfIt()
    {
        List<string> log = [];
        AttentionKernelRow[] rows = [Row("a", admits: false, runs: true, log), Row("b", true, true, log)];
        AttentionPlan plan = AttentionPlanner.Plan(rows, Call(DType.F32));
        Assert.Equal("b", plan.RowId);
        Assert.Equal("a (not admitted)", plan.Declined);
        Assert.Empty(log);
        Assert.StartsWith("b", plan.Describe());
    }
}
