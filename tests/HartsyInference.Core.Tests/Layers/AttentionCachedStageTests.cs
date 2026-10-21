using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The Flash-entry operator stage and the caller-provided output of the head-major stage: the one attention call each issues,
/// what it disposes, and what it refuses before any op, recorded without computing.</summary>
public sealed class AttentionCachedStageTests
{
    private static AttentionLayer Layer(int heads = 4, int kvHeads = 2, DType? compute = null, float scale = 0f) =>
        new(new AttentionLayerConfig { Heads = heads, KvHeads = kvHeads, HeadDim = 4, ComputeDType = compute, Scale = scale });

    private static Tensor Query(int tokens = 1, int heads = 4, int batch = 1, DType? dtype = null) =>
        new(new TensorShape(batch, heads, tokens, 4), dtype ?? DType.F32);

    private static Tensor Cache(int capacity = 8, int kvHeads = 2, int batch = 1) => new(new TensorShape(batch, kvHeads, capacity, 4), DType.F32);

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

    [Fact]
    public void HostPositions_IssueOneFlashCallThatCarriesTheWholeViewAndNothingElse()
    {
        AttentionTraceBackend backend = new();
        Tensor q = Query(tokens: 2), k = Cache(), v = Cache();

        Tensor attn = Layer().AttendCachedHeadMajor(backend, q, k, v, new AttentionCacheView(5, 3, Causal: true, SlidingWindow: 3, Softcap: 20f));

        Assert.Equal(["FlashAttention kv=5 group=2 causal=True qOffset=3 window=3 sinks=False"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(AttentionRoute.Flash, call.Route);
        Assert.Null(call.LegacyGap());
        Assert.Equal(new AttentionPositions(5, 3, 0, Bounded: true), call.Positions);
        Assert.False(call.Positions.OnDevice);
        Assert.True(call.Mask.Causal);
        Assert.Equal(2, call.Mask.WindowLeft);
        Assert.Equal(20f, call.Logits.Softcap);
        Assert.Null(call.Logits.Sinks);
        Assert.Equal(0.5f, call.Scale);
        Assert.Equal(2, call.KvGroup);
        Assert.Equal(new FlashCall(5, 2, true, 3, 0.5f, 20f, false, 3, 0), Assert.Single(backend.FlashCalls));
        Assert.Same(attn, call.Output.Storage);
        Assert.Same(q, call.Query.Storage);
        Assert.Same(k, call.Key.Storage);
        Assert.Same(v, call.Value.Storage);
        Assert.Equal(new TensorShape(1, 4, 2, 4), attn.Shape);
        Assert.Equal(8, call.Key.Tokens);
    }

    [Fact]
    public void Sinks_ReachTheCallAsTheLogitTerm()
    {
        AttentionTraceBackend backend = new();
        Tensor sinks = new(new TensorShape(4), DType.F32);

        Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(), new AttentionCacheView(6, 5, Causal: true, Sinks: sinks));

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Same(sinks, call.Logits.Sinks);
        Assert.Null(call.LegacyGap());
        Assert.True(Assert.Single(backend.FlashCalls).HasSinks);
    }

    [Fact]
    public void DevicePositions_BuildTheDeviceRouteAndCarryTheHostFallbackValues()
    {
        AttentionTraceBackend backend = new();

        Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(), new AttentionCacheView(4, 3, Causal: true, DevicePositions: 77));

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(AttentionRoute.FlashDevice, call.Route);
        Assert.Null(call.LegacyGap());
        Assert.Equal(new AttentionPositions(4, 3, 77, Bounded: true), call.Positions);
        Assert.Equal(["FlashAttentionDev kv=4 group=2 causal=True qOffset=3 window=0 dev=True"], backend.Ops);
        Assert.Equal(new FlashCall(4, 2, true, 3, 0.5f, 0f, false, 0, 77), Assert.Single(backend.FlashCalls));
    }

    [Fact]
    public void ANonCausalViewFromTheStartOfTheCache_IsTheCrossAttentionCall()
    {
        AttentionTraceBackend backend = new();

        Layer(heads: 4, kvHeads: 4).AttendCachedHeadMajor(backend, Query(), Cache(kvHeads: 4), Cache(kvHeads: 4), new AttentionCacheView(8, 0, Causal: false));

        Assert.Equal(new FlashCall(8, 1, false, 0, 0.5f, 0f, false, 0, 0), Assert.Single(backend.FlashCalls));
    }

    [Fact]
    public void ACustomScaleReachesTheCall()
    {
        AttentionTraceBackend backend = new();

        Layer(scale: 1f).AttendCachedHeadMajor(backend, Query(), Cache(), Cache(), new AttentionCacheView(1, 0, Causal: true));

        Assert.Equal(1f, Assert.Single(backend.Calls).Scale);
    }

    [Fact]
    public void TheOutputIsAllocatedInTheComputeDTypeElseTheQueryDType()
    {
        AttentionTraceBackend backend = new();
        AttentionCacheView view = new(1, 0, Causal: true);

        Tensor followed = Layer().AttendCachedHeadMajor(backend, Query(dtype: DType.F16), Cache(), Cache(), view);
        Tensor pinned = Layer(compute: DType.F32).AttendCachedHeadMajor(backend, Query(dtype: DType.F16), Cache(), Cache(), view);

        Assert.Equal(DType.F16, followed.DType);
        Assert.Equal(DType.F32, pinned.DType);
    }

    [Fact]
    public void ByDefaultOnlyTheQueryIsConsumed()
    {
        AttentionTraceBackend backend = new();
        Tensor q = Query(), k = Cache(), v = Cache();

        Tensor attn = Layer().AttendCachedHeadMajor(backend, q, k, v, new AttentionCacheView(1, 0, Causal: true));

        Assert.Equal([true, false, false, false], [IsDisposed(q), IsDisposed(k), IsDisposed(v), IsDisposed(attn)]);
    }

    [Theory]
    [InlineData(OperandOwnership.None, false, false, false)]
    [InlineData(OperandOwnership.Key | OperandOwnership.Value, false, true, true)]
    [InlineData(OperandOwnership.All, true, true, true)]
    public void EachFlagDisposesExactlyTheOperandsItNames(OperandOwnership consume, bool query, bool key, bool value)
    {
        AttentionTraceBackend backend = new();
        Tensor q = Query(), k = Cache(), v = Cache();

        Layer().AttendCachedHeadMajor(backend, q, k, v, new AttentionCacheView(1, 0, Causal: true), default, consume);

        Assert.Equal([query, key, value], [IsDisposed(q), IsDisposed(k), IsDisposed(v)]);
    }

    [Fact]
    public void AFailedCallDisposesTheOutputItAllocatedAndTheConsumedOperands()
    {
        AttentionTraceBackend backend = new() { FailAttention = true };
        Tensor q = Query(), k = Cache(), v = Cache();

        Assert.Throws<OutOfMemoryException>(() => Layer().AttendCachedHeadMajor(backend, q, k, v, new AttentionCacheView(1, 0, Causal: true)));

        Assert.True(IsDisposed(Assert.Single(backend.Calls).Output.Storage));
        Assert.Equal([true, false, false], [IsDisposed(q), IsDisposed(k), IsDisposed(v)]);
    }

    [Fact]
    public void ACallerProvidedOutputIsWrittenReturnedAndNeverDisposed()
    {
        AttentionTraceBackend backend = new();
        Tensor output = Query(tokens: 2);
        Tensor q = Query(tokens: 2);

        Tensor attn = Layer().AttendCachedHeadMajor(backend, q, Cache(), Cache(), new AttentionCacheView(3, 1, Causal: true), output: output);

        Assert.Same(output, attn);
        Assert.Same(output, Assert.Single(backend.Calls).Output.Storage);
        Assert.False(IsDisposed(output));
        Assert.True(IsDisposed(q));
    }

    [Fact]
    public void ACallerProvidedOutputSurvivesAFailedCall()
    {
        AttentionTraceBackend backend = new() { FailAttention = true };
        Tensor output = Query();

        Assert.Throws<OutOfMemoryException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(),
            new AttentionCacheView(1, 0, Causal: true), output: output));

        Assert.False(IsDisposed(output));
    }

    [Fact]
    public void AnOutputOfTheWrongShapeOrDTypeIsRefusedBeforeAnyOpAndDisposesNothing()
    {
        AttentionTraceBackend backend = new();
        Tensor q = Query(), k = Cache(), v = Cache();
        AttentionCacheView view = new(1, 0, Causal: true);

        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, q, k, v, view, output: Query(tokens: 2)));
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, q, k, v, view, output: Query(dtype: DType.F16)));

        Assert.Empty(backend.Ops);
        Assert.All((Tensor[])[q, k, v], t => Assert.False(IsDisposed(t)));
    }

    [Fact]
    public void TheProbeSeesTheOutputAfterTheCall_AndNothingElseIsAllowedInTheContext()
    {
        AttentionTraceBackend backend = new();
        List<string> points = new();
        AttentionCacheView view = new(1, 0, Causal: true);

        Tensor attn = Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(), view, new AttentionCallContext { Probe = new Probe(points) });

        Assert.Equal(["attention"], points);
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(), view,
            new AttentionCallContext { Bias = new Tensor(new TensorShape(1, 1, 1, 1), DType.F32) }));
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(), view,
            new AttentionCallContext { ReleaseInputs = true }));
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(), view,
            new AttentionCallContext { QueryRotation = new QkRotation(QkRotationKind.RotateHalf, null, null) }));
        Assert.Single(backend.Ops);
        Assert.False(IsDisposed(attn));
    }

    [Fact]
    public void DevicePositionsRefuseSinks_AndAViewBeyondTheCacheIsRefused_BeforeAnyOp()
    {
        AttentionTraceBackend backend = new();
        Tensor sinks = new(new TensorShape(4), DType.F32);

        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(),
            new AttentionCacheView(1, 0, Causal: true, Sinks: sinks, DevicePositions: 9)));
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(capacity: 4), Cache(capacity: 4),
            new AttentionCacheView(5, 0, Causal: true)));
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(heads: 2), Cache(), Cache(), new AttentionCacheView(1, 0, Causal: true)));
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(kvHeads: 4), Cache(kvHeads: 4), new AttentionCacheView(1, 0, Causal: true)));
        Assert.Throws<ArgumentException>(() => Layer().AttendCachedHeadMajor(backend, Query(), Cache(), Cache(capacity: 4), new AttentionCacheView(1, 0, Causal: true)));

        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void TheHeadMajorStageWritesACallerProvidedOutputAndNeverDisposesIt()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Layer(kvHeads: 4);
        Tensor output = Query(tokens: 3), q = Query(tokens: 3), k = Query(tokens: 5), v = Query(tokens: 5);

        Tensor attn = layer.AttendHeadMajorOperands(backend, q, k, v, default, OperandOwnership.Query, output);

        Assert.Same(output, attn);
        Assert.Equal(["Attention Dense heads=4/4 bounded=False"], backend.Ops);
        Assert.Same(output, Assert.Single(backend.Calls).Output.Storage);
        Assert.Equal([true, false, false, false], [IsDisposed(q), IsDisposed(k), IsDisposed(v), IsDisposed(output)]);
    }

    [Fact]
    public void TheHeadMajorStageLeavesAProvidedOutputAfterAFailureAndRefusesAMismatchedOne()
    {
        AttentionLayer layer = Layer(kvHeads: 4);
        Tensor output = Query(tokens: 3);
        AttentionTraceBackend failing = new() { FailAttention = true };

        Assert.Throws<OutOfMemoryException>(() => layer.AttendHeadMajorOperands(failing, Query(tokens: 3), Query(), Query(), default,
            OperandOwnership.All, output));
        Assert.False(IsDisposed(output));

        AttentionTraceBackend backend = new();
        Tensor q = Query(tokens: 3);
        Assert.Throws<ArgumentException>(() => layer.AttendHeadMajorOperands(backend, q, Query(), Query(), default, OperandOwnership.All, Query(tokens: 4)));
        Assert.Empty(backend.Ops);
        Assert.False(IsDisposed(q));
    }

    private sealed class Probe(List<string> points) : IAttentionProbe
    {
        public void Observe(string point, Tensor value) => points.Add(point);
    }
}
