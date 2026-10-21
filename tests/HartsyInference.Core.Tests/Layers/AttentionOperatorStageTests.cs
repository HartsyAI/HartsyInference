using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The head-major operator stage: the one attention call it issues, which operands it disposes, and what it refuses.</summary>
public sealed class AttentionOperatorStageTests
{
    private static AttentionLayer Layer(bool allowF16 = true, int heads = 2, int kvHeads = 2, DType? compute = null) =>
        new(new AttentionLayerConfig { Heads = heads, KvHeads = kvHeads, HeadDim = 4, AllowF16 = allowF16, ComputeDType = compute });

    private static Tensor Operand(int heads = 2, int tokens = 6, DType? dtype = null, int batch = 1) =>
        new(new TensorShape(batch, heads, tokens, 4), dtype ?? DType.F32);

    private static unsafe bool IsDisposed(Tensor t)
    {
        try
        {
            _ = (nint)t.DataPointer;
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    [Theory]
    [InlineData(true, "Attention Dense heads=2/2 bounded=True")]
    [InlineData(false, "Attention Dense heads=2/2 bounded=False")]
    public void IssuesOneDenseHeadMajorCallAndNothingElse(bool allowF16, string op)
    {
        AttentionTraceBackend backend = new();
        Tensor q = Operand(tokens: 3, batch: 2), k = Operand(tokens: 5, batch: 2), v = Operand(tokens: 5, batch: 2),
            bias = new(new TensorShape(1, 1, 3, 5), DType.F32);

        Tensor attn = Layer(allowF16).AttendHeadMajorOperands(backend, q, k, v, new AttentionCallContext { Bias = bias });

        Assert.Equal([op], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(AttentionRoute.Dense, call.Route);
        Assert.Equal(allowF16, call.Numerics.LegacyAssertedBounded);
        Assert.Null(call.LegacyGap());
        Assert.Equal(0.5f, call.Scale);
        Assert.Same(bias, call.Mask.Bias);
        Assert.Same(attn, call.Output.Storage);
        Assert.Same(q, call.Query.Storage);
        Assert.Same(k, call.Key.Storage);
        Assert.Same(v, call.Value.Storage);
        foreach (AttentionOperand operand in (AttentionOperand[])[call.Query, call.Key, call.Value, call.Output])
            Assert.Equal(AttentionLayout.HeadMajor, operand.Layout);
        Assert.Equal(new TensorShape(2, 2, 3, 4), attn.Shape);
        Assert.Equal(5, call.Key.Tokens);
    }

    [Fact]
    public void ACustomScaleReachesTheCall()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = 2, KvHeads = 2, HeadDim = 4, Scale = 0.125f });

        layer.AttendHeadMajorOperands(backend, Operand(), Operand(), Operand(), default);

        Assert.Equal(0.125f, Assert.Single(backend.Calls).Scale);
    }

    [Fact]
    public void TheOutputIsAllocatedInTheComputeDTypeElseTheQueryDType()
    {
        AttentionTraceBackend backend = new();

        Tensor followed = Layer().AttendHeadMajorOperands(backend, Operand(dtype: DType.F16), Operand(dtype: DType.F16),
            Operand(dtype: DType.F16), default);
        Tensor pinned = Layer(compute: DType.F32).AttendHeadMajorOperands(backend, Operand(dtype: DType.F16), Operand(dtype: DType.F16),
            Operand(dtype: DType.F16), default);

        Assert.Equal(DType.F16, followed.DType);
        Assert.Equal(DType.F32, pinned.DType);
    }

    [Theory]
    [InlineData(OperandOwnership.None, false, false, false)]
    [InlineData(OperandOwnership.Query, true, false, false)]
    [InlineData(OperandOwnership.Key, false, true, false)]
    [InlineData(OperandOwnership.Value, false, false, true)]
    [InlineData(OperandOwnership.Query | OperandOwnership.Value, true, false, true)]
    [InlineData(OperandOwnership.All, true, true, true)]
    public void EachFlagDisposesExactlyTheOperandsItNames(OperandOwnership consume, bool query, bool key, bool value)
    {
        AttentionTraceBackend backend = new();
        Tensor q = Operand(), k = Operand(), v = Operand();

        Tensor attn = Layer().AttendHeadMajorOperands(backend, q, k, v, default, consume);

        Assert.Equal([query, key, value], [IsDisposed(q), IsDisposed(k), IsDisposed(v)]);
        Assert.False(IsDisposed(attn));
    }

    [Fact]
    public void ConsumesEveryOperandByDefault()
    {
        AttentionTraceBackend backend = new();
        Tensor q = Operand(), k = Operand(), v = Operand();

        Layer().AttendHeadMajorOperands(backend, q, k, v, default);

        Assert.All((Tensor[])[q, k, v], t => Assert.True(IsDisposed(t)));
    }

    [Theory]
    [InlineData(OperandOwnership.None)]
    [InlineData(OperandOwnership.Key)]
    [InlineData(OperandOwnership.All)]
    public void AFailedCallDisposesTheOutputAndTheConsumedOperands(OperandOwnership consume)
    {
        AttentionTraceBackend backend = new() { FailAttention = true };
        Tensor q = Operand(), k = Operand(), v = Operand();

        Assert.Throws<OutOfMemoryException>(() => Layer().AttendHeadMajorOperands(backend, q, k, v, default, consume));

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.True(IsDisposed(call.Output.Storage));
        Assert.Equal([consume.HasFlag(OperandOwnership.Query), consume.HasFlag(OperandOwnership.Key), consume.HasFlag(OperandOwnership.Value)],
            [IsDisposed(q), IsDisposed(k), IsDisposed(v)]);
    }

    [Fact]
    public void ASuccessfulCallKeepsTheOutput()
    {
        AttentionTraceBackend backend = new();

        Tensor attn = Layer().AttendHeadMajorOperands(backend, Operand(), Operand(), Operand(), default);

        Assert.False(IsDisposed(attn));
    }

    [Fact]
    public void TheProbeSeesTheOutputAfterTheCall()
    {
        AttentionTraceBackend backend = new();
        List<string> points = new();
        Tensor? seen = null;

        Tensor attn = Layer().AttendHeadMajorOperands(backend, Operand(), Operand(), Operand(),
            new AttentionCallContext { Probe = new TraceProbe(points, t => seen = t) });

        Assert.Equal(["attention"], points);
        Assert.Same(attn, seen);
    }

    [Fact]
    public void ARefusedCallTouchesNothing()
    {
        AttentionTraceBackend backend = new();
        Tensor q = Operand(), k = Operand(), v = Operand();

        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, q, k, v, new AttentionCallContext { ReleaseInputs = true }));

        Assert.Empty(backend.Ops);
        Assert.All((Tensor[])[q, k, v], t => Assert.False(IsDisposed(t)));
    }

    [Fact]
    public void Rank3Operands_Throw()
    {
        AttentionTraceBackend backend = new();
        Tensor flat = new(new TensorShape(1, 6, 8), DType.F32);

        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, flat, Operand(), Operand(), default));
        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, Operand(), flat, Operand(), default));
        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, Operand(), Operand(), flat, default));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void OperandsThatDisagreeWithTheConfig_Throw()
    {
        AttentionTraceBackend backend = new();
        Tensor wideHead = new(new TensorShape(1, 2, 6, 8), DType.F32);

        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, Operand(heads: 3), Operand(), Operand(), default));
        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, Operand(), Operand(heads: 3), Operand(heads: 3), default));
        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, wideHead, Operand(), Operand(), default));
        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, Operand(), Operand(tokens: 5), Operand(tokens: 6), default));
        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, Operand(batch: 2), Operand(), Operand(), default));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void GroupedKeyValueHeads_AreNotSupported()
    {
        AttentionTraceBackend backend = new();

        Assert.Throws<NotSupportedException>(() => Layer(heads: 4, kvHeads: 2).AttendHeadMajorOperands(backend, Operand(4), Operand(2), Operand(2), default));
        Assert.Empty(backend.Ops);
    }

    public static TheoryData<string> RefusedContexts => new()
    {
        nameof(AttentionCallContext.QueryRotation), nameof(AttentionCallContext.KeyRotation), nameof(AttentionCallContext.Rotation),
        nameof(AttentionCallContext.HeadMajorRotation), nameof(AttentionCallContext.ValueRowScale), nameof(AttentionCallContext.KvRewrite),
        nameof(AttentionCallContext.KvCapture), nameof(AttentionCallContext.ReleaseInputs), nameof(AttentionCallContext.ReleaseQueryAfterProjection),
    };

    [Theory]
    [MemberData(nameof(RefusedContexts))]
    public void AContextFieldTheStageDoesNotApply_Throws(string field)
    {
        AttentionTraceBackend backend = new();
        Tensor table = new(new TensorShape(6, 4), DType.F32);
        AttentionCallContext context = field switch
        {
            nameof(AttentionCallContext.QueryRotation) => new() { QueryRotation = new(QkRotationKind.RotateHalf, table, table) },
            nameof(AttentionCallContext.KeyRotation) => new() { KeyRotation = new(QkRotationKind.InterleavedPairs, table, table, 6) },
            nameof(AttentionCallContext.Rotation) => new() { Rotation = new TraceRotation() },
            nameof(AttentionCallContext.HeadMajorRotation) => new() { HeadMajorRotation = new TraceRotation() },
            nameof(AttentionCallContext.ValueRowScale) => new() { ValueRowScale = Operand() },
            nameof(AttentionCallContext.KvRewrite) => new() { KvRewrite = new TraceRewrite() },
            nameof(AttentionCallContext.KvCapture) => new() { KvCapture = new CapturedKv() },
            nameof(AttentionCallContext.ReleaseInputs) => new() { ReleaseInputs = true },
            nameof(AttentionCallContext.ReleaseQueryAfterProjection) => new() { ReleaseQueryAfterProjection = true },
            _ => throw new ArgumentOutOfRangeException(nameof(field)),
        };

        Assert.Throws<ArgumentException>(() => Layer().AttendHeadMajorOperands(backend, Operand(), Operand(), Operand(), context));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void TheHeadMajorRouteOfAttend_StillRunsPermuteAttentionPermuteInThatOrder()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = 2, KvHeads = 2, HeadDim = 4, AllowF16 = true });
        List<string> points = new();
        Tensor Tm() => new(new TensorShape(1, 6, 2, 4), DType.F32);
        Tensor q = Tm(), k = Tm(), v = Tm();

        layer.Attend(backend, q, k, v, new TensorShape(6, 8), new AttentionCallContext { Probe = new TraceProbe(points, _ => { }) });

        Assert.Equal(["Permute0213 6,2,4", "Permute0213 6,2,4", "Permute0213 6,2,4", "Attention Dense heads=2/2 bounded=True", "Permute0213 2,6,4"],
            backend.Ops);
        Assert.Equal(["q", "k", "v", "attention"], points);
        Assert.All((Tensor[])[q, k, v], t => Assert.True(IsDisposed(t)));
    }

    private sealed class TraceProbe(List<string> points, Action<Tensor> onObserve) : IAttentionProbe
    {
        public void Observe(string point, Tensor value)
        {
            points.Add(point);
            onObserve(value);
        }
    }

    private sealed class TraceRotation : IQkRotation, IHeadMajorQkRotation
    {
        public void RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads) { }

        public void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens) { }
    }

    private sealed class TraceRewrite : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v) => (k, v);
    }
}
