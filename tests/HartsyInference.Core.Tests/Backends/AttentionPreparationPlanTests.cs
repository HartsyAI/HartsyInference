using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>A plan reports what its row expects to prepare and whether a later row would prepare less. The report never
/// changes which row is chosen: that is still the first row that covers and admits the call.</summary>
public sealed class AttentionPreparationPlanTests
{
    private sealed class Operands : IDisposable
    {
        public readonly Tensor Query = new(new TensorShape(1, 2, 3, 8), DType.F32);
        public readonly Tensor Key = new(new TensorShape(1, 2, 5, 8), DType.F32);
        public readonly Tensor Value = new(new TensorShape(1, 2, 5, 8), DType.F32);
        public readonly Tensor Output = new(new TensorShape(1, 2, 3, 8), DType.F32);

        public AttentionCall Call => AttentionCall.FromScaledDotProductAttention(Output, Query, Key, Value, null, 0.3f, allowF16: false);

        public void Dispose()
        {
            Query.Dispose();
            Key.Dispose();
            Value.Dispose();
            Output.Dispose();
        }
    }

    private static AttentionKernelRow Row(string id, AttentionLowering lowering, AttentionCostFunc? preparation, bool admits = true) =>
        new DelegateAttentionRow(new AttentionKernelCapability
        {
            Id = id, Entry = AttentionRoute.Dense, QueryTypes = AttentionDTypes.F32, KvTypes = AttentionDTypes.F32,
            Layouts = AttentionLayouts.HeadMajor, Lowering = lowering,
        }, null, (in AttentionCall _) => admits, static (in AttentionCall _) => true, preparation);

    [Fact]
    public void ARowWithNoLoweringStatesNoPreparation()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        AttentionPlan plan = AttentionPlanner.Plan([Row("native", AttentionLowering.None, null)], in call);
        Assert.Equal("native", plan.RowId);
        Assert.Equal(AttentionPreparationCost.None, plan.Preparation);
        Assert.Null(plan.Cheaper);
        Assert.Contains("prepares 0 B", plan.Describe());
    }

    [Fact]
    public void ARowThatDeclaresALoweringAndStatesNothingIsUnstated()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        AttentionPlan plan = AttentionPlanner.Plan([Row("casts", AttentionLowering.Cast, null)], in call);
        Assert.Null(plan.Preparation);
        Assert.Contains("prepares unstated", plan.Describe());
    }

    [Fact]
    public void TheFirstRowIsStillChosenWhenALaterRowWouldPrepareLess()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        AttentionKernelRow[] rows =
        [
            Row("casts", AttentionLowering.Cast, static (in AttentionCall _) => new AttentionPreparationCost(Cast: 960)),
            Row("masks", AttentionLowering.MaterializeMask, static (in AttentionCall _) => new AttentionPreparationCost(MaterializeMask: 60)),
            Row("native", AttentionLowering.None, null),
        ];
        AttentionPlan plan = AttentionPlanner.Plan(rows, in call);
        Assert.Equal("casts", plan.RowId);
        Assert.Equal(960, plan.Preparation!.Value.Total);
        // The least among the later rows is named, with its statement.
        Assert.Equal("native 0 B", plan.Cheaper);
        Assert.Contains("less preparation: native 0 B", plan.Describe());
        Assert.Same(rows[0], AttentionPlanner.Run(rows, in call));
    }

    [Fact]
    public void ALaterRowThatIsUnstatedOrNotAdmittedIsNeverReportedAsCheaper()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call;
        AttentionKernelRow[] rows =
        [
            Row("casts", AttentionLowering.Cast, static (in AttentionCall _) => new AttentionPreparationCost(Cast: 960)),
            Row("unstated", AttentionLowering.Permute, null),
            Row("refused", AttentionLowering.None, null, admits: false),
            Row("more", AttentionLowering.Cast, static (in AttentionCall _) => new AttentionPreparationCost(Cast: 2000)),
        ];
        Assert.Null(AttentionPlanner.Plan(rows, in call).Cheaper);
    }

    [Fact]
    public void AStatementCountsACastOnlyForOperandsInAnotherType()
    {
        using Operands operands = new();
        using Tensor halfKey = new(operands.Key.Shape, DType.F16);
        AttentionCall call = operands.Call with { Key = AttentionOperand.HeadMajor(halfKey) };
        // Computing in F32: only the F16 key is cast (80 elements to four bytes each).
        Assert.Equal(80 * sizeof(float), AttentionPreparationCost.CastsTo(in call, DType.F32));
        // Computing in F16: query and value are cast to two bytes each, and the F32 output is cast back.
        Assert.Equal((48 + 80) * sizeof(ushort) + 48 * sizeof(float), AttentionPreparationCost.CastsTo(in call, DType.F16));
    }
}
