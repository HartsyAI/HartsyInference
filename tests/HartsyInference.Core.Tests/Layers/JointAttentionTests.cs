using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The op sequences a joint attention issues: each stream's projection and norm, the concatenation in the
/// configured row order, one attention call over the joint rows and the split back, recorded without computing.</summary>
public sealed class JointAttentionTests
{
    private const int Heads = 4, HeadDim = 2, Inner = Heads * HeadDim, ImgRows = 6, TxtRows = 3;

    private static Tensor W(int rows) => new(new TensorShape(rows, Inner), DType.F32);

    /// <summary>Stream <paramref name="id"/> (1 or 2) names its weights 1xx or 2xx so a trace says which stream ran.</summary>
    private static AttentionLayer Stream(int id, AttentionQkNorm norm = AttentionQkNorm.PerHead, bool output = true) => new(
        new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, QkNorm = norm },
        new AttentionLayerWeights
        {
            Query = W(id * 100 + 1), Key = W(id * 100 + 2), Value = W(id * 100 + 3), Output = output ? W(id * 100 + 4) : null,
            QueryNorm = W(id * 100 + 6), KeyNorm = W(id * 100 + 7), QueryNormEps = 1e-6f, KeyNormEps = 1e-6f,
        });

    private static JointAttentionConfig Config(params int[] order) =>
        new() { Heads = Heads, HeadDim = HeadDim, ConcatOrder = order, AllowF16 = true };

    private static (Tensor Img, Tensor Txt) Inputs(int batch = 1) =>
        (new Tensor(new TensorShape(batch, ImgRows, Inner), DType.F32), new Tensor(new TensorShape(batch, TxtRows, Inner), DType.F32));

    private static JointAttention TokenMajorJoint() =>
        new(Config(1, 0) with { TokenMajorRoute = AttentionRoute.TokenMajor, HeadMajorForSage = true }, [Stream(1), Stream(2)]);

    private static int Permutes(AttentionTraceBackend backend) =>
        backend.Ops.Count(op => op.StartsWith("Permute0213", StringComparison.Ordinal));

    [Fact]
    public void ProjectAllThenNorm_TextFirst_ConcatsAttendsAndSplitsInRowOrder()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(1, 0), [Stream(1), Stream(2)]);
        (Tensor img, Tensor txt) = Inputs();
        Span<Tensor?> attended = [null, null];

        joint.Attend(backend, [img, txt], attended, new JointCallContext { ReleaseInputs = true });
        joint.ProjectOutput(backend, 0, attended[0]!);
        joint.ProjectOutput(backend, 1, attended[1]!);

        Assert.Equal(
        [
            "Linear w101", "Linear w102", "Linear w103", "Linear w201", "Linear w202", "Linear w203",
            "RmsNorm w106", "RmsNorm w107", "RmsNorm w206", "RmsNorm w207",
            "Concat rows 3+6", "Concat rows 3+6", "Concat rows 3+6",
            "Permute0213 9,4,2", "Permute0213 9,4,2", "Permute0213 9,4,2", "Attention Dense heads=4/4 bounded=True",
            "Permute0213 4,9,2", "SliceRows 3@0", "SliceRows 6@3", "Linear w104", "Linear w204",
        ], backend.Ops);
        Assert.Equal(new TensorShape(1, ImgRows, Inner), attended[0]!.Shape);
        Assert.Equal(new TensorShape(1, TxtRows, Inner), attended[1]!.Shape);
        Assert.Equal(new TensorShape(1, Heads, ImgRows + TxtRows, HeadDim), backend.Calls[0].Query.Storage.Shape);
    }

    [Fact]
    public void ProjectNormPerStream_ImageFirst_NormsEachStreamBeforeTheNextProjects()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(0, 1) with { StageOrder = JointStageOrder.ProjectNormPerStream },
            [Stream(1, AttentionQkNorm.FullWidthRms), Stream(2, AttentionQkNorm.FullWidthRms)]);
        (Tensor img, Tensor txt) = Inputs();
        Span<Tensor?> attended = [null, null];

        joint.Attend(backend, [img, txt], attended, new JointCallContext());

        Assert.Equal(
        [
            "Linear w101", "Linear w102", "Linear w103", "RmsNorm w106", "RmsNorm w107",
            "Linear w201", "Linear w202", "Linear w203", "RmsNorm w206", "RmsNorm w207",
            "Concat rows 6+3", "Concat rows 6+3", "Concat rows 6+3",
        ], backend.Ops.Take(13));
        Assert.Equal(["SliceRows 6@0", "SliceRows 3@6"], backend.Ops.Skip(backend.Ops.Count - 2));
    }

    [Fact]
    public void TheJointRotationRunsBeforeThePermutes_AndAHostRotationAfterThem()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(1, 0), [Stream(1), Stream(2)]);
        Span<Tensor?> attended = [null, null];

        (Tensor img, Tensor txt) = Inputs();
        joint.Attend(backend, [img, txt], attended,
            new JointCallContext { Joint = new AttentionCallContext { Rotation = new Rotation(backend.Ops) } });
        int rotated = backend.Ops.IndexOf("Rotate token-major 4/4");
        Assert.True(rotated > backend.Ops.LastIndexOf("Concat rows 3+6") && rotated < backend.Ops.IndexOf("Permute0213 9,4,2"));

        backend.Ops.Clear();
        (img, txt) = Inputs(batch: 2);
        joint.Attend(backend, [img, txt], attended,
            new JointCallContext { Joint = new AttentionCallContext { HeadMajorRotation = new Rotation(backend.Ops) } });
        int hostRotated = backend.Ops.IndexOf("Rotate head-major b=2 s=9");
        Assert.True(hostRotated > backend.Ops.LastIndexOf("Permute0213 9,4,2") && hostRotated < backend.Ops.IndexOf("Attention Dense heads=4/4 bounded=True"));
        Assert.Equal(["GatherRows 6", "GatherRows 12"], backend.Ops.Where(op => op.StartsWith("GatherRows", StringComparison.Ordinal)));
    }

    [Fact]
    public void ABatchedCall_CarriesTheBatchedAssertion()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(0, 1) with { AllowF16Batched = false }, [Stream(1), Stream(2)]);
        Span<Tensor?> attended = [null, null];

        (Tensor img, Tensor txt) = Inputs();
        joint.Attend(backend, [img, txt], attended, new JointCallContext());
        (img, txt) = Inputs(batch: 2);
        joint.Attend(backend, [img, txt], attended, new JointCallContext());

        Assert.Equal([true, false], backend.Calls.Select(c => c.Numerics.LegacyAssertedBounded));
    }

    /// <summary>A token-major joint config hands the concatenated <c>[1, S, heads·D]</c> rows to the token-major entry and splits
    /// its output, held in that shape, with no permute; a batch keeps the head-major call and its permutes.</summary>
    [Fact]
    public void ATokenMajorJointCall_AttendsTheConcatenatedRowsInPlace_ABatchStaysHeadMajor()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        JointAttention joint = TokenMajorJoint();
        (Tensor img, Tensor txt) = Inputs();
        Span<Tensor?> attended = [null, null];

        joint.Attend(backend, [img, txt], attended, new JointCallContext { ReleaseInputs = true });

        Assert.Equal(
        [
            "Concat rows 3+6", "Concat rows 3+6", "Concat rows 3+6", "Attention TokenMajor heads=4/4 bounded=True",
            "SliceRows 3@0", "SliceRows 6@3",
        ], backend.Ops.Skip(backend.Ops.IndexOf("Concat rows 3+6")));
        AttentionCall call = Assert.Single(backend.Calls);
        TensorShape joined = new(1, ImgRows + TxtRows, Inner);
        Assert.Equal(joined, call.Query.Storage.Shape);
        Assert.Equal(joined, call.Output.Storage.Shape);
        Assert.Equal(new TensorShape(1, ImgRows, Inner), attended[0]!.Shape);
        Assert.Equal(new TensorShape(1, TxtRows, Inner), attended[1]!.Shape);

        AttentionTraceBackend batched = new() { SupportsTokenMajorAttention = true };
        (img, txt) = Inputs(batch: 2);
        joint.Attend(batched, [img, txt], attended, new JointCallContext());
        Assert.Equal(AttentionRoute.Dense, Assert.Single(batched.Calls).Route);
        Assert.Equal(4, Permutes(batched));
    }

    /// <summary>Under a K/V rewrite (a context-parallel rank's rendezvous) the token-major joint call attends the keys the rewrite
    /// returns: the local query rows over the longer key rows.</summary>
    [Fact]
    public void ATokenMajorJointCall_AttendsTheKeysAKvRewriteReturns()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        (Tensor img, Tensor txt) = Inputs();
        Span<Tensor?> attended = [null, null];

        TokenMajorJoint().Attend(backend, [img, txt], attended,
            new JointCallContext { Joint = new AttentionCallContext { KvRewrite = new Prepend(4, backend.Ops) } });

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(AttentionRoute.TokenMajor, call.Route);
        Assert.Equal(ImgRows + TxtRows, call.QueryTokens);
        Assert.Equal(ImgRows + TxtRows + 4, call.KeyTokens);
        Assert.Equal(0, Permutes(backend));
        Assert.Equal(new TensorShape(1, ImgRows + TxtRows, Inner), call.Output.Storage.Shape);
    }

    [Fact]
    public void ABf16Grant_ReachesTheJointCall_AndExcludesEitherF16Assertion()
    {
        AttentionTraceBackend backend = new();
        JointAttentionConfig granted = Config(0, 1) with { AllowF16 = false, AllowBf16 = true };
        JointAttention joint = new(granted, [Stream(1), Stream(2)]);
        (Tensor img, Tensor txt) = Inputs();
        Span<Tensor?> attended = [null, null];

        joint.Attend(backend, [img, txt], attended, new JointCallContext());

        Assert.Equal(AttentionNumerics.Bf16Storage, Assert.Single(backend.Calls).Numerics);
        // The joint config refuses the pair itself, before any joint layer is built.
        Assert.Equal("config", Assert.Throws<ArgumentException>(() => new JointAttention(granted with { AllowF16 = true },
            [Stream(1), Stream(2)])).ParamName);
        Assert.Equal("config", Assert.Throws<ArgumentException>(() => new JointAttention(granted with { AllowF16Batched = true },
            [Stream(1), Stream(2)])).ParamName);
    }

    [Fact]
    public void AStreamWithoutAnOutputProjection_IsUsedUnprojected()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(1, 0), [Stream(1), Stream(2, output: false)]);
        (Tensor img, Tensor txt) = Inputs();
        Span<Tensor?> attended = [null, null];

        joint.Attend(backend, [img, txt], attended, new JointCallContext());

        Tensor unprojected = attended[1]!;
        Assert.Throws<InvalidOperationException>(() => joint.ProjectOutput(backend, 1, unprojected));
        joint.ProjectOutput(backend, 0, attended[0]!);
        Assert.Equal("Linear w104", backend.Ops[^1]);
    }

    [Fact]
    public void Config_RejectsStreamsThatCannotJoin()
    {
        Assert.Throws<ArgumentException>(() => new JointAttention(Config(0, 0), [Stream(1), Stream(2)]));
        Assert.Throws<ArgumentException>(() => new JointAttention(Config(0), [Stream(1)]));
        AttentionLayer gqa = new(new AttentionLayerConfig { Heads = Heads, KvHeads = 2, HeadDim = HeadDim });
        Assert.Throws<ArgumentException>(() => new JointAttention(Config(0, 1), [Stream(1), gqa]));
        AttentionLayer tokenMajor = new(new AttentionLayerConfig
        {
            Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, TokenMajorRoute = AttentionRoute.TokenMajor,
        });
        Assert.Throws<ArgumentException>(() => new JointAttention(Config(0, 1), [Stream(1), tokenMajor]));
    }

    [Fact]
    public void FlatOperands_AreRefusedWhereTheyWouldBeMisread()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer perHead = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, QkNorm = AttentionQkNorm.PerHead },
            new AttentionNormWeights(W(6), W(7), 1e-6f, 1e-6f));
        Tensor flat = new(new TensorShape(1, 5, Inner), DType.F32);
        Assert.Throws<NotSupportedException>(() => perHead.AttendFlat(backend, flat, flat, flat, flat.Shape, default));

        AttentionLayer plain = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim });
        Tensor narrow = new(new TensorShape(1, 5, Inner / 2), DType.F32);
        Assert.Throws<ArgumentException>(() => plain.AttendFlat(backend, narrow, narrow, narrow, narrow.Shape, default));
        Tensor table = new(new TensorShape(5, HeadDim), DType.F32);
        foreach (QkRotationKind kind in (QkRotationKind[])[QkRotationKind.RotateHalf, QkRotationKind.RotateHalfPaired])
        {
            QkRotation rotation = new(kind, table, table);
            Assert.Throws<NotSupportedException>(() => plain.AttendFlat(backend, flat, flat, flat, flat.Shape,
                new AttentionCallContext { QueryRotation = rotation, KeyRotation = rotation }));
        }
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void ABatchedF16JointCall_IsRefusedBeforeAnyOp()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(0, 1), [Stream(1), Stream(2)]);
        Tensor img = new(new TensorShape(2, ImgRows, Inner), DType.F16), txt = new(new TensorShape(2, TxtRows, Inner), DType.F16);

        Assert.Throws<NotSupportedException>(() => joint.Attend(backend, [img, txt], new Tensor?[2], new JointCallContext()));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void AFailedJointAttention_HandsBackNoPartialOutputs()
    {
        AttentionTraceBackend backend = new() { FailAttention = true };
        JointAttention joint = new(Config(1, 0), [Stream(1), Stream(2)]);
        (Tensor img, Tensor txt) = Inputs();
        Tensor?[] attended = [null, null];

        Assert.Throws<OutOfMemoryException>(() => joint.Attend(backend, [img, txt], attended, new JointCallContext()));
        Assert.All(attended, Assert.Null);
    }

    [Fact]
    public void APairedRotation_RunsOneRopeOverQAndK_AndNeedsTheSameTablesOnBoth()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Stream(1);
        Tensor x = new(new TensorShape(1, ImgRows, Inner), DType.F32), cos = new(new TensorShape(ImgRows, HeadDim), DType.F32);
        QkRotation paired = new(QkRotationKind.RotateHalfPaired, cos, cos);

        layer.Forward(backend, x, x, new AttentionCallContext { QueryRotation = paired, KeyRotation = paired });
        Assert.Single(backend.Ops, "ApplyRope");
        Assert.Equal(backend.Ops.IndexOf("RmsNorm w107") + 1, backend.Ops.IndexOf("ApplyRope"));

        backend.Ops.Clear();
        Assert.Throws<ArgumentException>(() => layer.Forward(backend, x, x, new AttentionCallContext { QueryRotation = paired }));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void ARotatedStream_RotatesItsOwnRowsAfterTheNormsAndBeforeTheConcat()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(0, 1), [Stream(1), Stream(2)]);
        (Tensor img, Tensor txt) = Inputs();
        Tensor table = new(new TensorShape(ImgRows, HeadDim), DType.F32);
        QkRotation pairs = new(QkRotationKind.InterleavedPairs, table, table, ImgRows);
        Span<Tensor?> attended = [null, null];

        joint.Attend(backend, [img, txt], attended, new JointCallContext
        {
            RotatedStream = 0, StreamRotation = new AttentionCallContext { QueryRotation = pairs, KeyRotation = pairs },
        });

        int norms = backend.Ops.IndexOf("RmsNorm w207"), concat = backend.Ops.IndexOf("Concat rows 6+3");
        Assert.Equal([norms + 1, norms + 2], backend.Ops.Select((op, i) => (op, i))
            .Where(x => x.op == "WanRopeInterleaved heads=4 d=2 seq=6").Select(x => x.i));
        Assert.Equal(norms + 3, concat);
    }

    [Fact]
    public void ARotatedStreamWithAJointRotation_IsRefusedBeforeAnyOp()
    {
        AttentionTraceBackend backend = new();
        JointAttention joint = new(Config(0, 1), [Stream(1), Stream(2)]);
        (Tensor img, Tensor txt) = Inputs();
        Tensor table = new(new TensorShape(ImgRows, HeadDim), DType.F32);
        QkRotation pairs = new(QkRotationKind.InterleavedPairs, table, table, ImgRows);
        JointCallContext both = new()
        {
            RotatedStream = 0, StreamRotation = new AttentionCallContext { QueryRotation = pairs, KeyRotation = pairs },
            Joint = new AttentionCallContext { QueryRotation = pairs, KeyRotation = pairs },
        };

        Assert.Throws<ArgumentException>(() => joint.Attend(backend, [img, txt], new Tensor?[2], both));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public unsafe void ACaptureWhoseAttentionThrows_GetsItsKvFreed()
    {
        AttentionTraceBackend backend = new() { FailAttention = true };
        AttentionLayer layer = Stream(1);
        Tensor x = new(new TensorShape(1, ImgRows, Inner), DType.F32);
        KvHolder capture = new();

        Assert.Throws<OutOfMemoryException>(() => layer.Forward(backend, x, x, new AttentionCallContext { KvCapture = capture }));
        Assert.Throws<ObjectDisposedException>(() => (nint)capture.Key!.DataPointer);
        Assert.Throws<ObjectDisposedException>(() => (nint)capture.Value!.DataPointer);
    }

    [Fact]
    public void AKvRewrite_ChangesTheKeyLength_AndACaptureKeepsTheRotatedKv()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Stream(1);
        Tensor x = new(new TensorShape(1, ImgRows, Inner), DType.F32);
        KvHolder capture = new();

        layer.Forward(backend, x, x, new AttentionCallContext { KvRewrite = new Prepend(2, backend.Ops), KvCapture = capture });

        Assert.Equal(["Rewrite", "Permute0213 6,4,2", "Permute0213 8,4,2", "Permute0213 8,4,2", "Attention Dense heads=4/4 bounded=False"],
            backend.Ops.Skip(backend.Ops.IndexOf("Rewrite")).Take(5));
        Assert.Equal(new TensorShape(1, ImgRows + 2, Heads, HeadDim), capture.Key!.Shape);
        Assert.Equal(new TensorShape(1, ImgRows + 2, Heads, HeadDim), capture.Value!.Shape);
    }

    [Fact]
    public unsafe void ACrossAttention_CanReleaseItsQueryInputBeforeKAndVExist()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = Stream(1);
        Tensor x = new(new TensorShape(1, ImgRows, Inner), DType.F32), text = new(new TensorShape(1, TxtRows, Inner), DType.F32);

        layer.Forward(backend, x, text, new AttentionCallContext { ReleaseQueryAfterProjection = true });

        Assert.Equal(["Linear w101", "Linear w102", "Linear w103", "RmsNorm w106", "RmsNorm w107", "Permute0213 6,4,2", "Permute0213 3,4,2"],
            backend.Ops.Take(7));
        Assert.Throws<ObjectDisposedException>(() => (nint)x.DataPointer);
        Assert.NotEqual(0, (nint)text.DataPointer);
        Assert.Throws<ArgumentException>(() => layer.Forward(backend, x, x, new AttentionCallContext { ReleaseQueryAfterProjection = true }));
        Assert.Throws<ArgumentException>(() => layer.Forward(backend, x, text,
            new AttentionCallContext { ReleaseQueryAfterProjection = true, ReleaseInputs = true }));
    }

    [Fact]
    public unsafe void AFailingKvRewrite_Propagates_AndKeepsTheKvItTookOver()
    {
        AttentionLayer layer = Stream(1);
        Tensor x = new(new TensorShape(1, ImgRows, Inner), DType.F32);
        Failing failing = new();
        Assert.Throws<InvalidOperationException>(() => layer.Forward(new AttentionTraceBackend(), x, x,
            new AttentionCallContext { KvRewrite = failing }));
        Assert.NotEqual(0, (nint)failing.Key!.DataPointer);
    }

    private sealed class Prepend(int rows, List<string> ops) : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
        {
            ops.Add("Rewrite");
            long[] dims = new long[k.Shape.Rank];
            for (int i = 0; i < dims.Length; i++) dims[i] = k.Shape[i];
            dims[1] += rows;
            TensorShape shape = new(dims);
            return (new Tensor(shape, k.DType), new Tensor(shape, v.DType));
        }
    }

    private sealed class Failing : IKvRewrite
    {
        public Tensor? Key;

        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
        {
            Key = k;
            throw new InvalidOperationException("peer aborted");
        }
    }

    private sealed class KvHolder : IKvCapture
    {
        public Tensor? Key, Value;

        public void Capture(Tensor k, Tensor v) => (Key, Value) = (k, v);
    }

    private sealed class Rotation(List<string> ops) : IQkRotation, IHeadMajorQkRotation
    {
        public void RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads) =>
            ops.Add($"Rotate token-major {heads}/{kvHeads}");

        public void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens) =>
            ops.Add($"Rotate head-major b={batch} s={tokens}");
    }
}
