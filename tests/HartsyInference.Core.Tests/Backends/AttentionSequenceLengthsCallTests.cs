using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>A call with a key length per sequence and an LSE output: what the contract accepts, which rows may take it,
/// and that the host reference runs it.</summary>
public sealed class AttentionSequenceLengthsCallTests
{
    private const int Batch = 2, Heads = 4, KvHeads = 2, Rows = 3, Capacity = 12, Dim = 16;

    private sealed class Operands : IDisposable
    {
        public readonly Tensor Query = Filled(new TensorShape(Batch, Heads, Rows, Dim), 11);
        public readonly Tensor Key = Filled(new TensorShape(Batch, KvHeads, Capacity, Dim), 12);
        public readonly Tensor Value = Filled(new TensorShape(Batch, KvHeads, Capacity, Dim), 13);
        public readonly Tensor Output = new(new TensorShape(Batch, Rows, Heads * Dim), DType.F32);
        public readonly Tensor Lse = new(new TensorShape(Batch, Heads, Rows), DType.F32);

        public AttentionCall Call(int[] lengths, bool causal = true, bool lse = true) =>
            AttentionCall.FromDecodeWithLengths(Output, Query, Key, Value, lengths, causal, 0.25f, lse ? Lse : null);

        public void Dispose()
        {
            foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[Query, Key, Value, Output, Lse]) tensor.Dispose();
        }

        private static Tensor Filled(TensorShape shape, int seed)
        {
            Tensor tensor = new(shape, DType.F32);
            Random random = new(seed);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
            return tensor;
        }
    }

    private static AttentionKernelCapability FlashRow(bool lengths, bool lse) => new()
    {
        Id = "test", Entry = AttentionRoute.Flash, QueryTypes = AttentionDTypes.F32, KvTypes = AttentionDTypes.F32,
        Layouts = AttentionLayouts.HeadMajor | AttentionLayouts.TokenMajor, StridedViews = true, Gqa = true,
        Masks = AttentionMaskSupport.Causal, SequenceLengths = lengths, Lse = lse,
    };

    [Fact]
    public void TheHostReferenceRunsTheCallAndWritesItsLse()
    {
        using Operands operands = new();
        int[] lengths = [7, 2];
        AttentionCall call = operands.Call(lengths);
        AttentionHostReference.Run(in call);

        using Tensor expected = new(operands.Output.Shape, DType.F32);
        float[] lse = new float[Batch * Heads * Rows];
        DecodeLengthsReference.Run(expected, operands.Query, operands.Key, operands.Value, lengths, causal: true, lse, scale: 0.25f);
        Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(operands.Output.AsReadOnlySpan<float>()));
        Assert.True(lse.AsSpan().SequenceEqual(operands.Lse.AsReadOnlySpan<float>()));
        // Sequence 1 holds two rows: under the causal rule its first query row sits below position zero.
        Assert.True(float.IsNegativeInfinity(lse[(1 * Heads + 0) * Rows + 0]));
        Assert.True(float.IsFinite(lse[(1 * Heads + 0) * Rows + 1]));
    }

    /// <summary>The CPU backend has no kernel for it either: it runs the contract reference, and says so.</summary>
    [Fact]
    public void TheCpuBackendRunsTheCallOnTheContractReference()
    {
        using Operands operands = new(), expected = new();
        int[] lengths = [7, 2];
        AttentionCall reference = expected.Call(lengths);
        AttentionHostReference.Run(in reference);

        using CpuBackend backend = new();
        AttentionCall call = operands.Call(lengths);
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
        backend.Attention(in call);
        Assert.Contains(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "cpu.host.reference");
        Assert.True(expected.Output.AsReadOnlySpan<float>().SequenceEqual(operands.Output.AsReadOnlySpan<float>()));
        Assert.True(expected.Lse.AsReadOnlySpan<float>().SequenceEqual(operands.Lse.AsReadOnlySpan<float>()));
        Assert.StartsWith("cpu.host.reference", backend.PlanAttention(in call).Reason);
    }

    [Fact]
    public void TheCallKeepsTheLengthsItWasBuiltWith()
    {
        using Operands operands = new(), expected = new();
        int[] lengths = [7, 2];
        AttentionCall call = operands.Call(lengths), reference = expected.Call([7, 2]);
        lengths[0] = 1;
        AttentionHostReference.Run(in call);
        AttentionHostReference.Run(in reference);
        Assert.True(expected.Output.AsReadOnlySpan<float>().SequenceEqual(operands.Output.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void ACallThatNamesItsLseTakesNoSecondOne()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call([7, 2]);
        Assert.Throws<ArgumentException>(() => AttentionHostReference.Run(in call, new float[Batch * Heads * Rows]));
    }

    [Fact]
    public void OnlyARowThatDeclaresThemTakesSequenceLengthsAndAnLse()
    {
        using Operands operands = new();
        AttentionCall both = operands.Call([7, 2]), lengthsOnly = operands.Call([7, 2], lse: false);
        Assert.Equal(AttentionExclusion.SequenceLengths, FlashRow(lengths: false, lse: true).Excludes(in both));
        Assert.Equal(AttentionExclusion.Lse, FlashRow(lengths: true, lse: false).Excludes(in both));
        Assert.Equal(AttentionExclusion.None, FlashRow(lengths: true, lse: true).Excludes(in both));
        Assert.Equal(AttentionExclusion.None, FlashRow(lengths: true, lse: false).Excludes(in lengthsOnly));
        // An LSE alone, on a call with one shared length, is its own request.
        AttentionCall lseOnly = both with { SequenceLengths = null };
        Assert.Equal(AttentionExclusion.Lse, FlashRow(lengths: true, lse: false).Excludes(in lseOnly));
        Assert.Equal(AttentionExclusion.None, FlashRow(lengths: false, lse: true).Excludes(in lseOnly));
    }

    [Fact]
    public void NoLegacyEntryOrFlashLoweringCarriesThem()
    {
        using Operands operands = new();
        AttentionCall both = operands.Call([7, 2]);
        Assert.Equal("per-sequence key lengths", both.LegacyGap());
        Assert.Equal("per-sequence key lengths", both.TermsGap());
        Assert.Equal("an LSE output", (both with { SequenceLengths = null }).LegacyGap());
        Assert.Equal("per-sequence key lengths", AttentionFlashLowering.Gap(both with { Route = AttentionRoute.Auto }));
        Assert.Equal("an LSE output", AttentionFlashLowering.Gap(both with { Route = AttentionRoute.Auto, SequenceLengths = null }));
    }

    [Fact]
    public void AnLseAloneIsWrittenForACallWithOneSharedLength()
    {
        using Operands operands = new();
        using Tensor output = new(operands.Query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromFlashAttention(output, operands.Query, operands.Key, operands.Value, 9, causal: true,
            qOffset: 6, scale: 0.25f, softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: null) with { Lse = operands.Lse };
        AttentionHostReference.Run(in call);
        float[] lse = new float[Batch * Heads * Rows];
        AttentionCall plain = call with { Lse = null };
        AttentionHostReference.Run(in plain, lse);
        Assert.True(lse.AsSpan().SequenceEqual(operands.Lse.AsReadOnlySpan<float>()));
        Assert.DoesNotContain(lse, float.IsNegativeInfinity);
    }

    [Fact]
    public void MalformedCallsAreRefused()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call([7, 2]);
        static void Refused<T>(AttentionCall bad) where T : Exception => Assert.Throws<T>(() => AttentionHostReference.Run(in bad));

        Refused<ArgumentException>(call with { SequenceLengths = [7] });
        Refused<ArgumentException>(call with { Positions = AttentionPositions.Host(7, 4) });
        Refused<ArgumentException>(call with { Positions = default(AttentionPositions) with { QueryOffset = 5 } });
        Refused<AttentionUnsupportedException>(call with { Mask = AttentionMask.FromFlash(true, 4) });
        Refused<AttentionUnsupportedException>(call with { Mask = call.Mask with { BlockSize = 2 } });
        Refused<AttentionUnsupportedException>(call with { Logits = new AttentionLogitTerms(30f, null, null) });
        using Tensor sinks = new(new TensorShape(Heads), DType.F32);
        Refused<AttentionUnsupportedException>(call with { Logits = new AttentionLogitTerms(0f, sinks, null) });
        using Tensor bias = new(new TensorShape(Rows, Capacity), DType.F32);
        Refused<AttentionUnsupportedException>(call with { Mask = call.Mask with { Bias = bias } });

        using Tensor wrongShape = new(new TensorShape(Batch, Rows, Heads), DType.F32);
        Refused<ArgumentException>(call with { Lse = wrongShape });
        using Tensor wrongType = new(operands.Lse.Shape, DType.F16);
        Refused<ArgumentException>(call with { Lse = wrongType });
        // Nothing was written by any of them.
        Assert.Equal(-1, operands.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(0f));

        using Tensor rank3 = new(new TensorShape(Batch, Rows, Heads * Dim), DType.F32);
        Assert.Throws<ArgumentException>(() => AttentionCall.FromDecodeWithLengths(operands.Output, rank3, operands.Key, operands.Value, [7, 2], true, 0.25f));
    }
}
