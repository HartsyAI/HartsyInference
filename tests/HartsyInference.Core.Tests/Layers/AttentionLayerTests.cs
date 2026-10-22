using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The op sequences the attention layer issues for each configuration the Wave A models use, recorded without
/// computing. The CPU goldens prove the numbers on the head-major path; these pin the order on every path, the token-major
/// ones included.</summary>
public sealed class AttentionLayerTests
{
    private static Tensor W(int rows) => new(new TensorShape(rows, 8), DType.F32);

    private static AttentionLayerWeights Weights(bool gate) => new()
    {
        Query = W(101), Key = W(102), Value = W(103), Output = W(104), Gate = gate ? W(105) : null,
        QueryNorm = W(106), KeyNorm = W(107), QueryNormEps = 1e-6f, KeyNormEps = 1e-6f,
    };

    private static AttentionLayerConfig Ltx(bool gated) => new()
    {
        Heads = 2, KvHeads = 2, HeadDim = 4, OutDim = 8, Projection = AttentionProjection.GroupedByInput,
        QkNorm = AttentionQkNorm.FullWidth, FullWidthNormEps = 1e-6f,
        Gate = gated ? AttentionGate.PerHeadDoubledSigmoid : AttentionGate.None, WidenKvToQuery = true,
        TokenMajorRoute = AttentionRoute.TokenMajor, AllowF16 = true,
    };

    private static AttentionLayerConfig Krea2(int heads = 4, int kvHeads = 2) => new()
    {
        Heads = heads, KvHeads = kvHeads, HeadDim = 4, QkNorm = AttentionQkNorm.PerHead, Gate = AttentionGate.Sigmoid,
        TokenMajorRoute = AttentionRoute.TokenMajorGqa, AllowF16 = true,
    };

    [Fact]
    public void Config_RejectsInconsistentGeometry()
    {
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Krea2(3, 2)));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Krea2() with { TokenMajorRoute = AttentionRoute.TokenMajor }));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Krea2() with { QkNorm = AttentionQkNorm.FullWidth }));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Ltx(gated: true) with { Projection = AttentionProjection.PerWeight }));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Krea2(), Weights(gate: false)));
    }

    [Fact]
    public void Ltx_TokenMajorSelfAttention_RunsTheFusedNormAndTheGatedProjection()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        AttentionLayer layer = new(Ltx(gated: true), Weights(gate: true));
        Tensor x = new(new TensorShape(6, 8), DType.F16), table = new(new TensorShape(6, 4), DType.F32);
        QkRotation rotation = new(QkRotationKind.FullWidthSplit, table, table);

        layer.Forward(backend, x, x, new AttentionCallContext { QueryRotation = rotation, KeyRotation = rotation });

        Assert.Equal(
        [
            "LinearMulti w105,w101,w102,w103", "NormRopeTokenMajor w106 rotated", "NormRopeTokenMajor w107 rotated",
            "Attention TokenMajor heads=2/2 bounded=True", "LinearHeadGated w104",
        ], backend.Ops);
        Assert.Equal(0.5f, backend.Calls[0].Scale);
    }

    [Fact]
    public void Ltx_HeadMajorCrossAttention_WidensAnF32KeyValueStream()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(Ltx(gated: false), Weights(gate: false));
        Tensor q = new(new TensorShape(6, 8), DType.F16), kv = new(new TensorShape(5, 8), DType.F32);

        layer.Forward(backend, q, kv, new AttentionCallContext());

        Assert.Equal(
        [
            "LinearMulti w101", "LinearMulti w102,w103", "NormRopeHeadMajor w106 unrotated",
            "CastToF16 [5, 8]", "NormRopeHeadMajor w107 unrotated", "CastToF16 [5, 8]", "Permute0213 5,2,4",
            "Attention Dense heads=2/2 bounded=True", "Permute0213 2,6,4", "Linear w104",
        ], backend.Ops);
    }

    [Fact]
    public void Ltx_TheSwitchIsReadOnEveryCall()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        AttentionLayer layer = new(Ltx(gated: false) with { TokenMajorKnob = EngineKnobs.Ltx2Tokenmajor }, Weights(gate: false));
        Tensor x = new(new TensorShape(6, 8), DType.F32);
        try
        {
            KnobStore.Set(EngineKnobs.Ltx2Tokenmajor, false);
            layer.Forward(backend, x, x, new AttentionCallContext());
            KnobStore.Set(EngineKnobs.Ltx2Tokenmajor, true);
            layer.Forward(backend, x, x, new AttentionCallContext());
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Ltx2Tokenmajor);
        }
        Assert.Equal([AttentionRoute.Dense, AttentionRoute.TokenMajor], backend.Calls.Select(c => c.Route));
    }

    [Fact]
    public void Krea2_GroupedTokenMajor_RotatesThroughTheModelThenGates()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorGqaAttention = true };
        AttentionLayer layer = new(Krea2(), Weights(gate: true));
        Tensor x = new(new TensorShape(1, 6, 16), DType.F16);

        layer.Forward(backend, x, x, new AttentionCallContext { Rotation = new TraceRotation(backend.Ops) });

        Assert.Equal(
        [
            "Linear w101", "Linear w102", "Linear w103", "Linear w105", "RmsNorm w106", "RmsNorm w107", "RotateTokenMajor 4/2",
            "Attention TokenMajorGqa heads=4/2 bounded=True", "Sigmoid", "Mul", "Linear w104",
        ], backend.Ops);
    }

    [Fact]
    public void Krea2_EqualHeads_KeepTheGroupedEntry()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorGqaAttention = true };
        new AttentionLayer(Krea2(2, 2), Weights(gate: true)).Forward(backend, new Tensor(new TensorShape(1, 6, 8), DType.F32),
            new Tensor(new TensorShape(1, 6, 8), DType.F32), new AttentionCallContext());

        Assert.Equal(AttentionRoute.TokenMajorGqa, Assert.Single(backend.Calls).Route);
    }

    [Fact]
    public void Krea2_WithoutGroupedTokenMajor_PermutesRepeatsAndPermutesBack()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(Krea2(), Weights(gate: true));
        Tensor x = new(new TensorShape(1, 6, 16), DType.F16);

        layer.Forward(backend, x, x, new AttentionCallContext());

        Assert.Equal(
        [
            "Linear w101", "Linear w102", "Linear w103", "Linear w105", "RmsNorm w106", "RmsNorm w107",
            "Permute0213 6,4,4", "Permute0213 6,2,4", "Permute0213 6,2,4", "RepeatKvHeads 2x2", "RepeatKvHeads 2x2",
            "Attention Dense heads=4/4 bounded=True", "Permute0213 4,6,4", "Sigmoid", "Mul", "Linear w104",
        ], backend.Ops);
    }

    [Fact]
    public void BatchedHostRotation_ForcesHeadMajorAndRunsAfterThePermutes()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorGqaAttention = true };
        AttentionLayer layer = new(Krea2(), Weights(gate: true));
        Tensor x = new(new TensorShape(2, 6, 16), DType.F32);

        layer.Forward(backend, x, x, new AttentionCallContext { HeadMajorRotation = new TraceRotation(backend.Ops) });

        Assert.Equal(
        [
            "Linear w101", "Linear w102", "Linear w103", "Linear w105", "RmsNorm w106", "RmsNorm w107",
            "Permute0213 6,4,4", "Permute0213 6,2,4", "Permute0213 6,2,4", "RotateHeadMajor b=2 4/2 s=6",
            "RepeatKvHeads 2x2", "RepeatKvHeads 2x2", "Attention Dense heads=4/4 bounded=True", "Permute0213 4,6,4",
            "Sigmoid", "Mul", "Linear w104",
        ], backend.Ops);
    }

    [Fact]
    public void ValueRowScale_MultipliesVBeforeTheGateProjection()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorGqaAttention = true };
        AttentionLayer layer = new(Krea2(), Weights(gate: true));
        Tensor x = new(new TensorShape(1, 6, 16), DType.F16);

        layer.Forward(backend, x, x, new AttentionCallContext { ValueRowScale = new Tensor(new TensorShape(1, 6, 2, 4), DType.F16) });
        Assert.Equal(["Linear w101", "Linear w102", "Linear w103", "Mul", "Linear w105"], backend.Ops.Take(5));

        backend.Ops.Clear();
        Assert.Throws<ArgumentException>(() => layer.Forward(backend, x, x,
            new AttentionCallContext { ValueRowScale = new Tensor(new TensorShape(1, 6, 4, 4), DType.F16) }));
        Assert.Empty(backend.Ops);   // refused before any projection
    }

    [Fact]
    public void ARotationTheNormCannotApply_IsRefusedBeforeAnyOp()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        Tensor x = new(new TensorShape(6, 8), DType.F16), table = new(new TensorShape(6, 4), DType.F32);
        AttentionLayer ltx = new(Ltx(gated: false), Weights(gate: false));
        QkRotation half = new(QkRotationKind.RotateHalf, table, table), split = new(QkRotationKind.FullWidthSplit, table, table);

        Assert.Throws<NotSupportedException>(() => ltx.Forward(backend, x, x, new AttentionCallContext { QueryRotation = half }));
        Assert.Throws<NotSupportedException>(() => ltx.Forward(backend, x, x,
            new AttentionCallContext { QueryRotation = split, KeyRotation = split, Rotation = new TraceRotation(new List<string>()) }));
        Assert.Throws<NotSupportedException>(() => ltx.Forward(backend, new Tensor(new TensorShape(2, 6, 8), DType.F16),
            new Tensor(new TensorShape(2, 6, 8), DType.F16), new AttentionCallContext { QueryRotation = split, KeyRotation = split }));
        AttentionLayer krea2 = new(Krea2(), Weights(gate: true));
        Assert.Throws<NotSupportedException>(() => krea2.Forward(backend, new Tensor(new TensorShape(1, 6, 16), DType.F16),
            new Tensor(new TensorShape(1, 6, 16), DType.F16), new AttentionCallContext { KeyRotation = split }));
        Assert.Empty(backend.Ops);
    }

    [Fact]
    public void Attend_RefusesOperandsWithoutABatchAxis()
    {
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = 4, KvHeads = 2, HeadDim = 4 });
        Assert.Throws<ArgumentException>(() => layer.Attend(new AttentionTraceBackend(), new Tensor(new TensorShape(6, 4, 4), DType.F32),
            new Tensor(new TensorShape(6, 2, 4), DType.F32), new Tensor(new TensorShape(6, 2, 4), DType.F32), new TensorShape(6, 16), default));
    }

    /// <summary>Forward hands its output projection the <c>[S, heads·D]</c> rows it has always read; Attend writes the caller's
    /// <c>[1, S, heads·D]</c> on the token-major route, the shape the head-major route returns.</summary>
    [Fact]
    public void EqualHeadTokenMajor_ForwardHoldsProjectionRows_AttendKeepsTheCallersShape()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = 2, KvHeads = 2, HeadDim = 4, TokenMajorRoute = AttentionRoute.TokenMajor },
            Weights(gate: false));
        Tensor x = new(new TensorShape(1, 6, 8), DType.F32);
        static Tensor Operand() => new(new TensorShape(1, 6, 2, 4), DType.F32);

        Tensor y = layer.Forward(backend, x, x, default);
        Tensor attended = layer.Attend(backend, Operand(), Operand(), Operand(), new TensorShape(1, 6, 8), default);
        AttentionTraceBackend headMajor = new();
        Tensor permuted = layer.Attend(headMajor, Operand(), Operand(), Operand(), new TensorShape(1, 6, 8), default);

        Assert.Equal(new TensorShape(6, 8), backend.Calls[0].Output.Storage.Shape);
        Assert.Equal(new TensorShape(1, 6, 8), y.Shape);
        Assert.Equal(AttentionRoute.TokenMajor, backend.Calls[1].Route);
        Assert.Same(attended, backend.Calls[1].Output.Storage);
        Assert.Equal(new TensorShape(1, 6, 8), attended.Shape);
        Assert.Equal(0, backend.Ops.Count(op => op.StartsWith("Permute0213", StringComparison.Ordinal)));
        Assert.Equal(AttentionRoute.Dense, Assert.Single(headMajor.Calls).Route);
        Assert.Equal(new TensorShape(1, 6, 8), permuted.Shape);
    }

    /// <summary>The CUDA head-major Sage row takes F16 at head dim 64 or 128, unmasked, with both sequences at the knob's
    /// minimum; the token-major entry has no Sage kernel, so a guarded layer keeps exactly those calls head-major.</summary>
    [Theory]
    [InlineData("sage-length", AttentionRoute.Dense)]
    [InlineData("sage-length-d64", AttentionRoute.Dense)]
    [InlineData("knob-minimum", AttentionRoute.Dense)]
    [InlineData("below-minimum", AttentionRoute.TokenMajor)]
    [InlineData("short-query", AttentionRoute.TokenMajor)]
    [InlineData("d96", AttentionRoute.TokenMajor)]
    [InlineData("f32", AttentionRoute.TokenMajor)]
    [InlineData("bias", AttentionRoute.TokenMajor)]
    [InlineData("sage-off", AttentionRoute.TokenMajor)]
    [InlineData("unguarded", AttentionRoute.TokenMajor)]
    public void HeadMajorForSage_KeepsTheCallsTheSageRowTakesHeadMajor(string name, AttentionRoute route)
    {
        int skv = name is "below-minimum" or "knob-minimum" ? 4096 : 8192;
        int sq = name == "short-query" ? 256 : skv;
        int headDim = name switch { "sage-length-d64" => 64, "d96" => 96, _ => 128 };
        DType act = name == "f32" ? DType.F32 : DType.F16;
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        AttentionLayer layer = new(new AttentionLayerConfig
        {
            Heads = 1, KvHeads = 1, HeadDim = headDim, TokenMajorRoute = AttentionRoute.TokenMajor, HeadMajorForSage = name != "unguarded",
            AllowF16 = true,
        }, Weights(gate: false));
        Tensor kv = new(new TensorShape(1, skv, 8), act), q = sq == skv ? kv : new(new TensorShape(1, sq, 8), act);
        Tensor? bias = name == "bias" ? new Tensor(new TensorShape(1, 1, 1, skv), DType.F32) : null;
        try
        {
            if (name == "sage-off") KnobStore.Set(EngineKnobs.SageAttn, false);
            if (name == "knob-minimum") KnobStore.Set(EngineKnobs.SageF16MinSkv, 4096);
            layer.Forward(backend, q, kv, new AttentionCallContext { Bias = bias });
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SageAttn);
            KnobStore.Clear(EngineKnobs.SageF16MinSkv);
        }

        Assert.Equal(route, Assert.Single(backend.Calls).Route);
        Assert.Equal(route == AttentionRoute.Dense ? 4 : 0, backend.Ops.Count(op => op.StartsWith("Permute0213", StringComparison.Ordinal)));
    }

    [Fact]
    public unsafe void AFailedAttentionCall_FreesTheOperandsItWasGiven()
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true, FailAttention = true };
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = 2, KvHeads = 2, HeadDim = 4, TokenMajorRoute = AttentionRoute.TokenMajor });
        Tensor q = new(new TensorShape(1, 6, 2, 4), DType.F32), k = new(new TensorShape(1, 6, 2, 4), DType.F32),
            v = new(new TensorShape(1, 6, 2, 4), DType.F32);

        Assert.Throws<OutOfMemoryException>(() => layer.Attend(backend, q, k, v, q.Shape, default));

        foreach (Tensor t in (Tensor[])[q, k, v])
            Assert.Throws<ObjectDisposedException>(() => (nint)t.DataPointer);
    }

    [Fact]
    public void TableRotations_RotateEachTensorWithItsOwnHeadCount()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(Krea2() with { Gate = AttentionGate.None }, Weights(gate: false));
        Tensor x = new(new TensorShape(1, 6, 16), DType.F32), table = new(new TensorShape(6, 4), DType.F32);
        QkRotation pairs = new(QkRotationKind.InterleavedPairs, table, table, 6);

        layer.Forward(backend, x, x, new AttentionCallContext { QueryRotation = pairs, KeyRotation = pairs });

        Assert.Equal(["WanRopeInterleaved heads=4 d=4 seq=6", "WanRopeInterleaved heads=2 d=4 seq=6"],
            backend.Ops.Where(op => op.StartsWith("WanRope", StringComparison.Ordinal)));
    }

    [Fact]
    public void Attend_RunsTheProjectedOperandsOfAModelsOwnProjection()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = 4, KvHeads = 2, HeadDim = 4, AllowF16 = true });
        Tensor table = new(new TensorShape(6, 4), DType.F32);
        QkRotation half = new(QkRotationKind.RotateHalf, table, table);

        Tensor flat = layer.Attend(backend, new Tensor(new TensorShape(1, 6, 4, 4), DType.F32), new Tensor(new TensorShape(1, 6, 2, 4), DType.F32),
            new Tensor(new TensorShape(1, 6, 2, 4), DType.F32), new TensorShape(6, 16),
            new AttentionCallContext { QueryRotation = half, KeyRotation = half });

        Assert.Equal(
        [
            "ApplyRopeSingle", "ApplyRopeSingle", "Permute0213 6,4,4", "Permute0213 6,2,4", "Permute0213 6,2,4",
            "RepeatKvHeads 2x2", "RepeatKvHeads 2x2", "Attention Dense heads=4/4 bounded=True", "Permute0213 4,6,4",
        ], backend.Ops);
        Assert.Equal(new TensorShape(6, 16), flat.Shape);
    }

    [Fact]
    public void Probe_SeesEachIntermediateInOrder()
    {
        AttentionTraceBackend backend = new();
        List<string> points = new();
        AttentionLayer layer = new(Krea2(4, 4) with { Gate = AttentionGate.None, TokenMajorRoute = AttentionRoute.Dense }, Weights(gate: false));
        Tensor x = new(new TensorShape(1, 6, 16), DType.F32);

        layer.Forward(backend, x, x, new AttentionCallContext { Probe = new TraceProbe(points) });

        Assert.Equal(["q", "k", "v", "qNormed", "kNormed", "attention"], points);
    }

    private sealed class TraceRotation(List<string> ops) : IQkRotation, IHeadMajorQkRotation
    {
        public void RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads) => ops.Add($"RotateTokenMajor {heads}/{kvHeads}");

        public void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens) =>
            ops.Add($"RotateHeadMajor b={batch} {heads}/{kvHeads} s={tokens}");
    }

    private sealed class TraceProbe(List<string> points) : IAttentionProbe
    {
        public void Observe(string point, Tensor value) => points.Add(point);
    }
}
