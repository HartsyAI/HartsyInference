using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The shared-query branch stages: the op order of a cache miss and hit, which tensors each stage disposes, and who owns
/// cached or stored key/value. Recorded without computing.</summary>
public sealed class BranchStageTests
{
    private const int Heads = 2, HeadDim = 4, Tokens = 3, CtxTokens = 5;

    private static Tensor W(int rows) => new(new TensorShape(rows, 8), DType.F32);

    private static AttentionLayer Layer(bool perHeadNorm = false, bool allowF16 = true) => new(
        new AttentionLayerConfig
        {
            Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, AllowF16 = allowF16,
            QkNorm = perHeadNorm ? AttentionQkNorm.PerHead : AttentionQkNorm.None,
        },
        new AttentionLayerWeights
        {
            Query = W(101), Key = W(102), Value = W(103), Output = W(104), QueryNorm = W(106), KeyNorm = W(107),
            QueryNormEps = 1e-6f, KeyNormEps = 1e-6f,
        });

    private static Tensor Hidden(int tokens = Tokens, int batch = 1) => new(new TensorShape(batch, tokens, 8), DType.F32);

    private static Tensor Mh(int tokens, int batch = 1) => new(new TensorShape(batch, Heads, tokens, HeadDim), DType.F32);

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
    public void AMissRunsTheQueryStageThenKeyValueThenOneAttention()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Layer(perHeadNorm: true);

        Tensor qMh = layer.ProjectQueryHeadMajor(backend, Hidden(), releaseInput: false, default);
        Tensor attn = layer.AttendBranchHeadMajor(backend, qMh, Hidden(CtxTokens), default);

        Assert.Equal(["Linear w101", "RmsNorm w106", "Permute0213 3,2,4",
            "Linear w102", "Linear w103", "RmsNorm w107", "Permute0213 5,2,4", "Permute0213 5,2,4",
            "Attention Dense heads=2/2 bounded=True"], backend.Ops);
        Assert.Equal(new TensorShape(1, Heads, Tokens, HeadDim), qMh.Shape);
        Assert.Equal(new TensorShape(1, Heads, Tokens, HeadDim), attn.Shape);
    }

    [Fact]
    public void WithoutANormTheStagesIssueNoNormOp()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Layer();

        Tensor qMh = layer.ProjectQueryHeadMajor(backend, Hidden(), releaseInput: false, default);
        layer.AttendBranchHeadMajor(backend, qMh, Hidden(CtxTokens), default);

        Assert.Equal(["Linear w101", "Permute0213 3,2,4", "Linear w102", "Linear w103", "Permute0213 5,2,4", "Permute0213 5,2,4",
            "Attention Dense heads=2/2 bounded=True"], backend.Ops);
    }

    [Fact]
    public void TheTablesRotateAfterTheNormAndBeforeThePermute()
    {
        AttentionTraceBackend backend = new();
        Tensor table = new(new TensorShape(CtxTokens, HeadDim), DType.F32);
        AttentionCallContext context = new()
        {
            QueryRotation = new(QkRotationKind.InterleavedPairs, table, table, Tokens),
            KeyRotation = new(QkRotationKind.InterleavedPairs, table, table, CtxTokens),
        };
        AttentionLayer layer = Layer(perHeadNorm: true);

        Tensor qMh = layer.ProjectQueryHeadMajor(backend, Hidden(), releaseInput: false, in context);
        layer.AttendBranchHeadMajor(backend, qMh, Hidden(CtxTokens), in context);

        Assert.Equal(["Linear w101", "RmsNorm w106", "WanRopeInterleaved heads=2 d=4 seq=3", "Permute0213 3,2,4",
            "Linear w102", "Linear w103", "RmsNorm w107", "WanRopeInterleaved heads=2 d=4 seq=5", "Permute0213 5,2,4", "Permute0213 5,2,4",
            "Attention Dense heads=2/2 bounded=True"], backend.Ops);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheQueryInputIsDisposedOnlyWhenReleased(bool release)
    {
        AttentionTraceBackend backend = new();
        Tensor input = Hidden();

        Tensor qMh = Layer().ProjectQueryHeadMajor(backend, input, release, default);

        Assert.Equal(release, IsDisposed(input));
        Assert.False(IsDisposed(qMh));
    }

    [Fact]
    public void AMissWithoutACacheDisposesKeyValueAndBorrowsTheQueryAndTheInput()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Layer();
        Tensor qMh = Mh(Tokens), context = Hidden(CtxTokens);

        Tensor attn = layer.AttendBranchHeadMajor(backend, qMh, context, default);

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Same(qMh, call.Query.Storage);
        Assert.True(IsDisposed(call.Key.Storage));
        Assert.True(IsDisposed(call.Value.Storage));
        Assert.False(IsDisposed(qMh));
        Assert.False(IsDisposed(context));
        Assert.False(IsDisposed(attn));
        Assert.Same(attn, call.Output.Storage);
    }

    [Fact]
    public void ABranchCallPinsTheLegacyAttentionNumerics()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Layer(allowF16: true);

        layer.AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), default);

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(AttentionRoute.Dense, call.Route);
        Assert.True(call.Numerics.LegacyAssertedBounded);
        Assert.Null(call.LegacyGap());
        Assert.Equal(1.0f / MathF.Sqrt(HeadDim), call.Scale);
        Assert.Null(call.Mask.Bias);
        Assert.Equal(CtxTokens, call.Key.Tokens);
    }

    [Fact]
    public void MergeHeadsPermutesBackAndDisposesTheHeadMajorOutput()
    {
        AttentionTraceBackend backend = new();
        Tensor attn = Mh(Tokens, batch: 2);

        Tensor flat = Layer().MergeHeads(backend, attn, new TensorShape(2, Tokens, 8));

        Assert.Equal(["Permute0213 2,3,4"], backend.Ops);
        Assert.Equal(new TensorShape(2, Tokens, 8), flat.Shape);
        Assert.Equal(DType.F32, flat.DType);
        Assert.True(IsDisposed(attn));
        Assert.False(IsDisposed(flat));
    }

    [Fact]
    public void ACachedHitIsBorrowedAndNeverDisposed()
    {
        AttentionTraceBackend backend = new();
        FakeCache cache = new(backend) { Hit = (Mh(CtxTokens), Mh(CtxTokens)) };
        Tensor qMh = Mh(Tokens), context = Hidden(CtxTokens);

        Layer().AttendBranchHeadMajor(backend, qMh, context, new AttentionCallContext { CachedKv = cache });

        Assert.Equal(["Attention Dense heads=2/2 bounded=True"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Same(cache.Hit!.Value.Key, call.Key.Storage);
        Assert.Same(cache.Hit.Value.Value, call.Value.Storage);
        Assert.False(IsDisposed(cache.Hit.Value.Key));
        Assert.False(IsDisposed(cache.Hit.Value.Value));
        Assert.Equal([(backend, context)], cache.Gets);
        Assert.Empty(cache.Stores);
    }

    [Fact]
    public void AMissAsksTheCacheBeforeProjectingAndOffersWhatItProjected()
    {
        AttentionTraceBackend backend = new();
        FakeCache cache = new(backend) { Accept = true };
        Tensor context = Hidden(CtxTokens);

        Layer().AttendBranchHeadMajor(backend, Mh(Tokens), context, new AttentionCallContext { CachedKv = cache });

        Assert.Equal([0], cache.GetOpCounts);
        Assert.Equal([4], cache.StoreOpCounts);
        AttentionCall call = Assert.Single(backend.Calls);
        (IBackend storedBackend, Tensor storedInput, Tensor storedKey, Tensor storedValue) = Assert.Single(cache.Stores);
        Assert.Same(backend, storedBackend);
        Assert.Same(context, storedInput);
        Assert.Same(call.Key.Storage, storedKey);
        Assert.Same(call.Value.Storage, storedValue);
    }

    [Fact]
    public void KeyValueTheCacheTookAreNotDisposed()
    {
        AttentionTraceBackend backend = new();
        FakeCache cache = new(backend) { Accept = true };

        Layer().AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), new AttentionCallContext { CachedKv = cache });

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.False(IsDisposed(call.Key.Storage));
        Assert.False(IsDisposed(call.Value.Storage));
    }

    [Fact]
    public void KeyValueTheCacheDeclinedAreDisposedAfterTheCall()
    {
        AttentionTraceBackend backend = new();
        FakeCache cache = new(backend) { Accept = false };

        Layer().AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), new AttentionCallContext { CachedKv = cache });

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Single(cache.Stores);
        Assert.True(IsDisposed(call.Key.Storage));
        Assert.True(IsDisposed(call.Value.Storage));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AFailedAttentionDisposesTheOutputAndOnlyKeyValueNobodyOwns(bool accept)
    {
        AttentionTraceBackend backend = new() { FailAttention = true };
        FakeCache cache = new(backend) { Accept = accept };
        Tensor qMh = Mh(Tokens);

        Assert.Throws<OutOfMemoryException>(() => Layer().AttendBranchHeadMajor(backend, qMh, Hidden(CtxTokens),
            new AttentionCallContext { CachedKv = cache }));

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.True(IsDisposed(call.Output.Storage));
        Assert.Equal(!accept, IsDisposed(call.Key.Storage));
        Assert.Equal(!accept, IsDisposed(call.Value.Storage));
        Assert.False(IsDisposed(qMh));
    }

    [Fact]
    public void TheSelfPathThroughForwardIssuesTheProjectionsThenQueryKeyValuePermutes()
    {
        AttentionTraceBackend backend = new();
        Tensor normed = Hidden();

        Tensor projected = Layer().Forward(backend, normed, normed, new AttentionCallContext { ReleaseInputs = true });

        Assert.Equal(["Linear w101", "Linear w102", "Linear w103", "Permute0213 3,2,4", "Permute0213 3,2,4", "Permute0213 3,2,4",
            "Attention Dense heads=2/2 bounded=True", "Permute0213 2,3,4", "Linear w104"], backend.Ops);
        Assert.True(IsDisposed(normed));
        Assert.Equal(new TensorShape(1, Tokens, 8), projected.Shape);
    }

    [Fact]
    public void AGroupedOrGatedOrFusedLayerCannotRunBranchStages()
    {
        AttentionTraceBackend backend = new();
        AttentionLayerWeights weights = new() { Query = W(101), Key = W(102), Value = W(103), Output = W(104), Gate = W(105) };
        AttentionLayer gated = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, Gate = AttentionGate.Sigmoid }, weights);
        AttentionLayer grouped = new(new AttentionLayerConfig
        {
            Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, Projection = AttentionProjection.GroupedByInput,
        }, weights);

        Assert.Throws<NotSupportedException>(() => gated.ProjectQueryHeadMajor(backend, Hidden(), false, default));
        Assert.Throws<NotSupportedException>(() => grouped.ProjectQueryHeadMajor(backend, Hidden(), false, default));
        Assert.Throws<NotSupportedException>(() => gated.AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), default));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void ALayerWithoutWeightsCannotRunBranchStages()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer bare = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim });

        Assert.Throws<InvalidOperationException>(() => bare.ProjectQueryHeadMajor(backend, Hidden(), false, default));
        Assert.Throws<InvalidOperationException>(() => bare.AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), default));
    }

    [Fact]
    public void GroupedKeyValueHeadsCannotAttendABranch()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer gqa = new(new AttentionLayerConfig { Heads = 4, KvHeads = 2, HeadDim = HeadDim },
            new AttentionLayerWeights { Query = W(101), Key = W(102), Value = W(103) });

        Assert.Throws<NotSupportedException>(() => gqa.AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), default));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void AQueryThatDisagreesWithTheKeyValueInputOrTheConfigThrows()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Layer();

        Assert.Throws<ArgumentException>(() => layer.AttendBranchHeadMajor(backend, Mh(Tokens, batch: 2), Hidden(CtxTokens), default));
        Assert.Throws<ArgumentException>(() => layer.AttendBranchHeadMajor(backend,
            new Tensor(new TensorShape(1, 3, Tokens, HeadDim), DType.F32), Hidden(CtxTokens), default));
        Assert.Throws<ArgumentException>(() => layer.AttendBranchHeadMajor(backend, Hidden(), Hidden(CtxTokens), default));
        Assert.Empty(backend.Ops);
    }

    public static TheoryData<string> RefusedContexts => new()
    {
        nameof(AttentionCallContext.Rotation), nameof(AttentionCallContext.HeadMajorRotation), "PairedQuery", "PairedKey",
        nameof(AttentionCallContext.ValueRowScale), nameof(AttentionCallContext.KvRewrite), nameof(AttentionCallContext.KvCapture),
        nameof(AttentionCallContext.ReleaseInputs), nameof(AttentionCallContext.ReleaseQueryAfterProjection),
    };

    [Theory]
    [MemberData(nameof(RefusedContexts))]
    public void AContextFieldABranchStageDoesNotApply_ThrowsBeforeAnyOp(string field)
    {
        AttentionTraceBackend backend = new();
        Tensor table = new(new TensorShape(CtxTokens, HeadDim), DType.F32);
        AttentionCallContext context = field switch
        {
            nameof(AttentionCallContext.Rotation) => new() { Rotation = new NoRotation() },
            nameof(AttentionCallContext.HeadMajorRotation) => new() { HeadMajorRotation = new NoRotation() },
            "PairedQuery" => new() { QueryRotation = new(QkRotationKind.RotateHalfPaired, table, table) },
            "PairedKey" => new() { KeyRotation = new(QkRotationKind.RotateHalfPaired, table, table) },
            nameof(AttentionCallContext.ValueRowScale) => new() { ValueRowScale = Mh(Tokens) },
            nameof(AttentionCallContext.KvRewrite) => new() { KvRewrite = new NoRewrite() },
            nameof(AttentionCallContext.KvCapture) => new() { KvCapture = new CapturedKv() },
            nameof(AttentionCallContext.ReleaseInputs) => new() { ReleaseInputs = true },
            nameof(AttentionCallContext.ReleaseQueryAfterProjection) => new() { ReleaseQueryAfterProjection = true },
            _ => throw new ArgumentOutOfRangeException(nameof(field)),
        };
        AttentionLayer layer = Layer();

        Assert.ThrowsAny<Exception>(() => layer.ProjectQueryHeadMajor(backend, Hidden(), false, in context));
        Assert.ThrowsAny<Exception>(() => layer.AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), in context));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void ACachedKvIsRefusedByEveryStageThatHasNoBranch()
    {
        AttentionTraceBackend backend = new();
        AttentionCallContext context = new() { CachedKv = new FakeCache(backend) };
        AttentionLayer layer = Layer();
        Tensor Tm() => new(new TensorShape(1, Tokens, Heads, HeadDim), DType.F32);

        Assert.Throws<ArgumentException>(() => layer.Forward(backend, Hidden(), Hidden(), in context));
        Assert.Throws<ArgumentException>(() => layer.Attend(backend, Tm(), Tm(), Tm(), new TensorShape(Tokens, 8), in context));
        Assert.Throws<ArgumentException>(() => layer.AttendHeadMajorOperands(backend, Mh(Tokens), Mh(Tokens), Mh(Tokens), in context));
        Assert.Empty(backend.Ops);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheConfigsOfTheUNetAndControlNetPinTheirLegacyAllowF16(bool allowF16)
    {
        AttentionTraceBackend backend = new();

        Layer(allowF16: allowF16).AttendBranchHeadMajor(backend, Mh(Tokens), Hidden(CtxTokens), default);

        Assert.Equal(allowF16, Assert.Single(backend.Calls).Numerics.LegacyAssertedBounded);
    }

    private sealed class FakeCache(AttentionTraceBackend backend) : ICachedKv
    {
        public (Tensor Key, Tensor Value)? Hit { get; init; }

        public bool Accept { get; init; }

        public List<(IBackend, Tensor)> Gets { get; } = new();

        public List<int> GetOpCounts { get; } = new();

        public List<(IBackend, Tensor, Tensor, Tensor)> Stores { get; } = new();

        public List<int> StoreOpCounts { get; } = new();

        public bool TryGet(IBackend backend1, Tensor kvInput, out Tensor keyMh, out Tensor valueMh)
        {
            Gets.Add((backend1, kvInput));
            GetOpCounts.Add(backend.Ops.Count);
            (keyMh, valueMh) = Hit ?? (null!, null!);
            return Hit is not null;
        }

        public bool TryStore(IBackend backend1, Tensor kvInput, Tensor keyMh, Tensor valueMh)
        {
            Stores.Add((backend1, kvInput, keyMh, valueMh));
            StoreOpCounts.Add(backend.Ops.Count);
            return Accept;
        }
    }

    private sealed class NoRotation : IQkRotation, IHeadMajorQkRotation
    {
        public void RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads) { }

        public void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens) { }
    }

    private sealed class NoRewrite : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v) => (k, v);
    }
}
