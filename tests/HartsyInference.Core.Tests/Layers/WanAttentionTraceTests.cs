using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The Wan-family attention configs (self, cross with a key/value-only image branch, S2V injector, Matrix-Game 3 action
/// module) over rank-2 <c>[S, dim]</c> rows: the op order, the rotation kinds, the tap and rewrite hooks, and the numerics pins.
/// Recorded without computing.</summary>
public sealed class WanAttentionTraceTests
{
    private const int Heads = 2, HeadDim = 4, Dim = Heads * HeadDim, Tokens = 3, CtxTokens = 5;

    private static Tensor W(int rows) => new(new TensorShape(rows, Dim), DType.F32);

    private static AttentionLayerConfig Config(bool allowF16 = true) => new()
    {
        Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, QkNorm = AttentionQkNorm.FullWidthRms, ComputeDType = DType.F32,
        AllowF16 = allowF16,
    };

    private static AttentionLayer Self(AttentionFormats refused = AttentionFormats.None) => new(Config() with { RefusedFormats = refused },
        new AttentionLayerWeights
        {
            Query = W(101), Key = W(102), Value = W(103), Output = W(104), QueryNorm = W(106), KeyNorm = W(107),
            QueryNormEps = 1e-6f, KeyNormEps = 1e-6f,
        });

    private static AttentionLayer ImageBranch() => new(Config(), new AttentionLayerWeights
    {
        Key = W(112), Value = W(113), KeyNorm = W(117), KeyNormEps = 1e-6f,
    });

    private static Tensor Rows(int rows) => new(new TensorShape(rows, Dim), DType.F32);

    private static Tensor Table(int rank) => rank == 2
        ? new Tensor(new TensorShape(Tokens, HeadDim), DType.F32)
        : new Tensor(new TensorShape(Heads, Tokens, HeadDim), DType.F32);

    private static AttentionCallContext Rotated(QkRotationKind kind, Tensor table)
    {
        QkRotation rotation = new(kind, table, table, Tokens);
        return new AttentionCallContext { QueryRotation = rotation, KeyRotation = rotation };
    }

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
    public void SelfAttentionOverRank2RowsRunsTheLegacyOrder()
    {
        AttentionTraceBackend backend = new();
        Tensor x = Rows(Tokens), mask = new(new TensorShape(1, 1, Tokens, Tokens), DType.F32);
        AttentionCallContext context = Rotated(QkRotationKind.InterleavedPairs, Table(2)) with { Bias = mask };

        Tensor output = Self().Forward(backend, x, x, in context);

        Assert.Equal(["Linear w101", "Linear w102", "Linear w103", "RmsNorm w106", "RmsNorm w107",
            "WanRopeInterleaved heads=2 d=4 seq=3", "WanRopeInterleaved heads=2 d=4 seq=3",
            "Permute0213 3,2,4", "Permute0213 3,2,4", "Permute0213 3,2,4",
            "Attention Dense heads=2/2 bounded=True", "Permute0213 2,3,4", "Linear w104"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Same(mask, call.Mask.Bias);
        Assert.Equal(new TensorShape(Tokens, Dim), output.Shape);
        Assert.Equal(DType.F32, output.DType);
        Assert.False(IsDisposed(x));
    }

    [Fact]
    public void ARank3TableRotatesEachTensorWithItsPerHeadKernel()
    {
        AttentionTraceBackend backend = new();
        Tensor x = Rows(Tokens);

        Self().Forward(backend, x, x, Rotated(QkRotationKind.InterleavedPairsPerHead, Table(3)));

        Assert.Equal(["Linear w101", "Linear w102", "Linear w103", "RmsNorm w106", "RmsNorm w107",
            "WanRopeInterleavedPerHead heads=2 d=4 seq=3", "WanRopeInterleavedPerHead heads=2 d=4 seq=3",
            "Permute0213 3,2,4", "Permute0213 3,2,4", "Permute0213 3,2,4",
            "Attention Dense heads=2/2 bounded=True", "Permute0213 2,3,4", "Linear w104"], backend.Ops);
    }

    [Fact]
    public void ThePerHeadKindIsRefusedByTheFusedFullWidthNorm()
    {
        AttentionLayer fused = new(Config() with { QkNorm = AttentionQkNorm.FullWidth }, new AttentionLayerWeights
        {
            Query = W(101), Key = W(102), Value = W(103), Output = W(104), QueryNorm = W(106), KeyNorm = W(107),
        });
        Tensor x = Rows(Tokens);

        Assert.Throws<NotSupportedException>(() =>
            fused.Forward(new AttentionTraceBackend(), x, x, Rotated(QkRotationKind.InterleavedPairsPerHead, Table(3))));
    }

    private sealed class RecordingTap(AttentionTraceBackend backend) : IQkvTap
    {
        public List<string>? OpsAtCall { get; private set; }

        public int Calls { get; private set; }

        public Tensor[] Seen { get; private set; } = [];

        public bool[] DisposedAtCall { get; private set; } = [];

        public void Observe(IBackend b, Tensor q, Tensor k, Tensor v)
        {
            Calls++;
            OpsAtCall = [.. backend.Ops];
            Seen = [q, k, v];
            DisposedAtCall = [IsDisposed(q), IsDisposed(k), IsDisposed(v)];
        }
    }

    [Fact]
    public void TheTapFiresOnceAfterTheNormAndBeforeTheRotationWithoutTakingTheTensors()
    {
        AttentionTraceBackend backend = new();
        RecordingTap tap = new(backend);
        Tensor x = Rows(Tokens);
        AttentionCallContext context = Rotated(QkRotationKind.InterleavedPairs, Table(2)) with { Tap = tap };

        Self().Forward(backend, x, x, in context);

        Assert.Equal(1, tap.Calls);
        Assert.Equal(["Linear w101", "Linear w102", "Linear w103", "RmsNorm w106", "RmsNorm w107"], tap.OpsAtCall);
        Assert.Equal([false, false, false], tap.DisposedAtCall);
        Assert.All(tap.Seen, t => Assert.Equal(new TensorShape(Tokens, Dim), t.Shape));
        Assert.All(tap.Seen, t => Assert.True(IsDisposed(t)));
        Assert.Equal(2, backend.Ops.Count(op => op.StartsWith("WanRopeInterleaved", StringComparison.Ordinal)));
    }

    [Fact]
    public void ATapRunsOnlyOnTheTransformStages()
    {
        AttentionTraceBackend backend = new();
        RecordingTap tap = new(backend);
        AttentionCallContext context = new() { Tap = tap };

        Assert.Throws<ArgumentException>(() => Self().ProjectQueryHeadMajor(backend, Rows(Tokens), releaseInput: false, in context));
        Assert.Throws<ArgumentException>(() => Self().AttendBranchHeadMajor(backend,
            new Tensor(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32), Rows(CtxTokens), in context));
        Assert.Throws<ArgumentException>(() => Self().AttendHeadMajorOperands(backend,
            new Tensor(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32), new Tensor(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32),
            new Tensor(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32), in context));
        Assert.Equal(0, tap.Calls);
        Assert.Empty(backend.Ops);
    }

    private sealed class ExchangeRewrite(int rows) : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
        {
            k.Dispose();
            v.Dispose();
            return (Rows(rows), Rows(rows));
        }
    }

    [Fact]
    public void AfterARewriteTheKeyValueTokenCountComesFromTheRank2Layout()
    {
        AttentionTraceBackend backend = new();
        Tensor x = Rows(Tokens);
        AttentionCallContext context = Rotated(QkRotationKind.InterleavedPairs, Table(2)) with { KvRewrite = new ExchangeRewrite(CtxTokens) };

        Tensor output = Self().Forward(backend, x, x, in context);

        Assert.Equal(["Linear w101", "Linear w102", "Linear w103", "RmsNorm w106", "RmsNorm w107",
            "WanRopeInterleaved heads=2 d=4 seq=3", "WanRopeInterleaved heads=2 d=4 seq=3",
            "Permute0213 3,2,4", "Permute0213 5,2,4", "Permute0213 5,2,4",
            "Attention Dense heads=2/2 bounded=True", "Permute0213 2,3,4", "Linear w104"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(CtxTokens, call.Key.Tokens);
        Assert.Equal(Tokens, call.Query.Tokens);
        Assert.Equal(new TensorShape(Tokens, Dim), output.Shape);
    }

    private sealed class ThrowingRewrite : IKvRewrite
    {
        public Tensor? K { get; private set; }

        public Tensor? V { get; private set; }

        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
        {
            (K, V) = (k, v);
            throw new InvalidOperationException("Context-parallel peer rank failed — aborting this rank's forward.");
        }
    }

    [Fact]
    public void AThrowingExchangeKeepsItsKeyValueAndTheLayerRunsNoAttention()
    {
        AttentionTraceBackend backend = new();
        ThrowingRewrite rewrite = new();
        Tensor x = Rows(Tokens);
        AttentionCallContext context = Rotated(QkRotationKind.InterleavedPairs, Table(2)) with { KvRewrite = rewrite };

        Assert.Throws<InvalidOperationException>(() => Self().Forward(backend, x, x, in context));

        Assert.NotNull(rewrite.K);
        Assert.False(IsDisposed(rewrite.K!));
        Assert.False(IsDisposed(rewrite.V!));
        Assert.False(IsDisposed(x));
        Assert.DoesNotContain(backend.Ops, op => op.StartsWith("Attention", StringComparison.Ordinal));
    }

    [Fact]
    public void CrossAttentionSharesOneQueryAcrossTheTextAndImageBranches()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer text = Self(), image = ImageBranch();
        Tensor qInput = Rows(Tokens), textRows = Rows(CtxTokens), imageRows = Rows(4);
        TensorShape flatShape = new(Tokens, Dim);

        Tensor qMh = text.ProjectQueryHeadMajor(backend, qInput, releaseInput: false, default);
        Tensor flatText = text.MergeHeads(backend, text.AttendBranchHeadMajor(backend, qMh, textRows, default), flatShape);
        Tensor flatImage = image.MergeHeads(backend, image.AttendBranchHeadMajor(backend, qMh, imageRows, default), flatShape);

        Assert.Equal(["Linear w101", "RmsNorm w106", "Permute0213 3,2,4",
            "Linear w102", "Linear w103", "RmsNorm w107", "Permute0213 5,2,4", "Permute0213 5,2,4",
            "Attention Dense heads=2/2 bounded=True", "Permute0213 2,3,4",
            "Linear w112", "Linear w113", "RmsNorm w117", "Permute0213 4,2,4", "Permute0213 4,2,4",
            "Attention Dense heads=2/2 bounded=True", "Permute0213 2,3,4"], backend.Ops);
        Assert.False(IsDisposed(qMh));
        Assert.False(IsDisposed(qInput));
        Assert.False(IsDisposed(textRows));
        Assert.False(IsDisposed(imageRows));
        Assert.Equal(flatShape, flatText.Shape);
        Assert.Equal(flatShape, flatImage.Shape);
        Assert.All(backend.Calls, c => Assert.Same(qMh, c.Query.Storage));
    }

    [Fact]
    public void AKeyValueOnlyWeightSetNeedsOnlyTheKeyNormAndHasNoQueryStage()
    {
        AttentionLayer image = ImageBranch();

        Assert.Throws<InvalidOperationException>(() => image.ProjectQueryHeadMajor(new AttentionTraceBackend(), Rows(Tokens), false, default));
        Assert.Throws<InvalidOperationException>(() => image.ProjectSelf(new AttentionTraceBackend(), Rows(Tokens), false));
        Assert.Throws<InvalidOperationException>(() => image.Forward(new AttentionTraceBackend(), Rows(Tokens), Rows(Tokens), default));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Config(), new AttentionLayerWeights
        {
            Key = W(112), Value = W(113),
        }));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Config(), new AttentionLayerWeights
        {
            Query = W(101), Key = W(112), Value = W(113), KeyNorm = W(117),
        }));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Config(), new AttentionLayerWeights
        {
            Query = W(101), Key = W(112), Value = W(113), QueryNorm = W(106),
        }));
    }

    [Fact]
    public void AnAudioInjectorFrameAttendsRank2OperandsAsOneBatchItem()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(Config(allowF16: false) with { QkNorm = AttentionQkNorm.None });
        Tensor q = Rows(Tokens), k = Rows(CtxTokens), v = Rows(CtxTokens);

        Tensor frame = layer.AttendFlat(backend, q, k, v, new TensorShape(Tokens, Dim), default);

        Assert.Equal(["Permute0213 3,2,4", "Permute0213 5,2,4", "Permute0213 5,2,4",
            "Attention Dense heads=2/2 bounded=False", "Permute0213 2,3,4"], backend.Ops);
        Assert.Equal(new TensorShape(Tokens, Dim), frame.Shape);
        Assert.True(IsDisposed(q));
        Assert.True(IsDisposed(k));
        Assert.True(IsDisposed(v));
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(new TensorShape(1, Heads, Tokens, HeadDim), call.Query.Storage.Shape);
        Assert.Equal(CtxTokens, call.Key.Tokens);
    }

    [Fact]
    public void FlatOperandsOfMixedRankOrWidthAreRefused()
    {
        AttentionLayer layer = new(Config() with { QkNorm = AttentionQkNorm.None });
        TensorShape flat = new(Tokens, Dim);

        Assert.Throws<ArgumentException>(() => layer.AttendFlat(new AttentionTraceBackend(), Rows(Tokens),
            new Tensor(new TensorShape(1, CtxTokens, Dim), DType.F32), new Tensor(new TensorShape(1, CtxTokens, Dim), DType.F32), flat, default));
        Assert.Throws<ArgumentException>(() => layer.AttendFlat(new AttentionTraceBackend(), Rows(Tokens),
            new Tensor(new TensorShape(CtxTokens, Dim + 1), DType.F32), new Tensor(new TensorShape(CtxTokens, Dim + 1), DType.F32), flat, default));
    }

    [Fact]
    public void AMatrixGame3ActionStreamIsOneUnmaskedCallThatTakesAllThreeOperands()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(Config() with { QkNorm = AttentionQkNorm.None });
        Tensor Operand() => new(new TensorShape(4, Heads, Tokens, HeadDim), DType.F32);
        Tensor q = Operand(), k = Operand(), v = Operand();

        Tensor attn = layer.AttendHeadMajorOperands(backend, q, k, v, default);

        Assert.Equal(["Attention Dense heads=2/2 bounded=True"], backend.Ops);
        Assert.True(IsDisposed(q));
        Assert.True(IsDisposed(k));
        Assert.True(IsDisposed(v));
        Assert.False(IsDisposed(attn));
        Assert.Equal(new TensorShape(4, Heads, Tokens, HeadDim), attn.Shape);
        Assert.Null(Assert.Single(backend.Calls).Mask.Bias);
    }

    [Theory]
    [InlineData("self", true)]
    [InlineData("cross", true)]
    [InlineData("self-int8-refused", true)]
    [InlineData("cross-int8-refused", true)]
    [InlineData("s2v", false)]
    [InlineData("mg3", true)]
    public void EachWanConfigPinsTheLegacyAllowF16OnADenseRouteWithNoGap(string site, bool allowF16)
    {
        AttentionTraceBackend backend = new();
        Tensor Mh(int tokens) => new(new TensorShape(1, Heads, tokens, HeadDim), DType.F32);
        AttentionFormats refused = site.EndsWith("-int8-refused", StringComparison.Ordinal) ? AttentionFormats.Int8 : AttentionFormats.None;
        switch (site)
        {
            case "self" or "self-int8-refused":
            {
                Tensor x = Rows(Tokens);
                Self(refused).Forward(backend, x, x, default);
                break;
            }
            case "cross" or "cross-int8-refused":
                Self(refused).AttendBranchHeadMajor(backend, Mh(Tokens), Rows(CtxTokens), default);
                break;
            case "s2v":
                new AttentionLayer(Config(allowF16: false) with { QkNorm = AttentionQkNorm.None })
                    .AttendFlat(backend, Rows(Tokens), Rows(CtxTokens), Rows(CtxTokens), new TensorShape(Tokens, Dim), default);
                break;
            default:
                new AttentionLayer(Config() with { QkNorm = AttentionQkNorm.None })
                    .AttendHeadMajorOperands(backend, Mh(Tokens), Mh(Tokens), Mh(Tokens), default);
                break;
        }

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(AttentionRoute.Dense, call.Route);
        Assert.Equal(allowF16, call.Numerics.LegacyAssertedBounded);
        Assert.Equal(refused, call.Numerics.Refused);
        Assert.Null(call.LegacyGap());
        Assert.Equal(1.0f / MathF.Sqrt(HeadDim), call.Scale);
        Assert.Equal(DType.F32, call.Output.Storage.DType);
    }
}
