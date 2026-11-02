using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>A model whose values exceed F16's range (Lens's un-normalised V) demands F32 or BF16. The demand refuses F16 rows and
/// F16 operands whatever the call asserts, and where F16 storage overflows, the demanded routes stay finite.</summary>
public sealed class AttentionWideRangeTests
{
    private const int Rows = 4, D = 8;
    private const float LensScaleV = 1e5f;   // past F16's 65504: stored as F16 it is +Inf

    private static AttentionKernelCapability Cap(string id) => new()
    {
        Id = id, Entry = AttentionRoute.Dense, QueryTypes = AttentionDTypes.F32 | AttentionDTypes.F16 | AttentionDTypes.BF16,
        KvTypes = AttentionDTypes.F32 | AttentionDTypes.F16 | AttentionDTypes.BF16, Layouts = AttentionLayouts.HeadMajor,
    };

    private static AttentionCall Call(DType qk, DType v, DType o, AttentionNumerics numerics)
    {
        Tensor q = new(new TensorShape(1, 1, Rows, D), qk), k = new(new TensorShape(1, 1, Rows, D), qk);
        Tensor value = new(new TensorShape(1, 1, Rows, D), v), output = new(new TensorShape(1, 1, Rows, D), o);
        return AttentionCall.FromScaledDotProductAttention(output, q, k, value, null, 1f, allowF16: false) with { Numerics = numerics };
    }

    private static AttentionCall Call(DType dt, AttentionNumerics numerics) => Call(dt, dt, dt, numerics);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheDemand_RefusesAnF16RowThatAdmits_WhateverTheLegacyAssertion(bool legacy)
    {
        List<string> log = [];
        AttentionKernelRow f16 = new DelegateAttentionRow(Cap("f16") with { UsesQueryKey = AttentionFormats.F16, UsesProbValue = AttentionFormats.F16 },
            "f16", static (in AttentionCall _) => true, (in AttentionCall _) => { log.Add("f16"); return true; });
        AttentionKernelRow f32 = new DelegateAttentionRow(Cap("f32"), "f32", static (in AttentionCall _) => true,
            (in AttentionCall _) => { log.Add("f32"); return true; });
        AttentionCall call = Call(DType.F32, AttentionNumerics.WideRange(bf16Storage: true) with { LegacyAssertedBounded = legacy });

        AttentionKernelRow[] rows = [f16, f32];
        Assert.Equal(AttentionExclusion.RefusedFormat, f16.Capability.Excludes(in call));
        Assert.Equal("f32", AttentionPlanner.Run(rows, in call).Capability.Id);
        Assert.Equal(["f32"], log);
    }

    [Fact]
    public void ARowThatMultipliesInItsOperandFormat_IsRefusedForAnF16Operand_AndOnlyUnderTheDemand()
    {
        AttentionKernelCapability operandRow = Cap("naive") with { ComputesInOperandFormat = true };
        AttentionCall f16 = Call(DType.F16, AttentionNumerics.WideRange(bf16Storage: true));
        AttentionCall f16Plain = Call(DType.F16, AttentionNumerics.Exact);
        AttentionCall f32 = Call(DType.F32, AttentionNumerics.WideRange(bf16Storage: true));

        Assert.Equal(AttentionExclusion.RefusedFormat, operandRow.Excludes(in f16));
        Assert.Equal(AttentionExclusion.None, operandRow.Excludes(in f16Plain));
        Assert.Equal(AttentionExclusion.None, operandRow.Excludes(in f32));
    }

    [Fact]
    public void AnF16OperandOverflows_WhereTheDemandedRoutesStayFinite()
    {
        using Tensor v32 = new(new TensorShape(1, 1, Rows, D), DType.F32);
        v32.AsSpan<float>().Fill(LensScaleV);
        using Tensor v16 = v32.CastTo(DType.F16), vBf16 = v32.CastTo(DType.BF16);
        using Tensor q = new(new TensorShape(1, 1, Rows, D), DType.F32), k = new(new TensorShape(1, 1, Rows, D), DType.F32);
        q.AsSpan<float>().Clear();
        k.AsSpan<float>().Clear();

        // The F16 route: V stored F16, every entry +Inf, so every output row is non-finite.
        using Tensor outF16 = new(new TensorShape(1, 1, Rows, D), DType.F32);
        AttentionCall f16Route = AttentionCall.FromScaledDotProductAttention(outF16, q, k, v16, null, 1f, allowF16: true);
        AttentionHostReference.Run(in f16Route);
        Assert.Contains(outF16.AsSpan<float>().ToArray(), x => !float.IsFinite(x));

        // The demanded F32 route: uniform attention over equal V rows returns V itself.
        using Tensor outF32 = new(new TensorShape(1, 1, Rows, D), DType.F32);
        AttentionCall f32Route = AttentionCall.FromScaledDotProductAttention(outF32, q, k, v32, null, 1f, allowF16: false)
            with { Numerics = AttentionNumerics.WideRange(bf16Storage: true) };
        AttentionHostReference.Run(in f32Route);
        Assert.All(outF32.AsSpan<float>().ToArray(), x => Assert.InRange(x, LensScaleV * 0.999f, LensScaleV * 1.001f));

        // The demanded BF16 route: BF16 keeps F32's range, so the same V stays finite.
        using Tensor outBf16 = new(new TensorShape(1, 1, Rows, D), DType.F32);
        AttentionCall bf16Route = AttentionCall.FromScaledDotProductAttention(outBf16, q, k, vBf16, null, 1f, allowF16: false)
            with { Numerics = AttentionNumerics.WideRange(bf16Storage: true) };
        AttentionHostReference.Run(in bf16Route);
        Assert.All(outBf16.AsSpan<float>().ToArray(), x => Assert.InRange(x, LensScaleV * 0.99f, LensScaleV * 1.01f));

        // A request for the F16 route under the demand is refused, not clipped.
        AttentionCall refused = AttentionCall.FromScaledDotProductAttention(outF16, q, k, v16, null, 1f, allowF16: true)
            with { Numerics = AttentionNumerics.WideRange(bf16Storage: true) };
        Assert.Throws<AttentionUnsupportedException>(() => AttentionHostReference.Run(in refused));
    }
}
