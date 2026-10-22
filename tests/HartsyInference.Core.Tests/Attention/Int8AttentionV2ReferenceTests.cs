using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Attention;

/// <summary>INT8 attention v2's executable specification (<see cref="Int8AttentionV2Reference"/>): its arithmetic pins, its
/// agreement with the contract oracle, and the adversarial cases with the trade-off each variant makes on them.</summary>
public sealed class Int8AttentionV2ReferenceTests(ITestOutputHelper output)
{
    private static readonly Int8AttentionV2Options Base = new();
    private static readonly Int8AttentionV2Options F16Pv = new() { Probabilities = Int8ProbabilityFormat.F16 };
    private static readonly Int8AttentionV2Options Fp8Pv = new() { Probabilities = Int8ProbabilityFormat.Fp8 };
    private static readonly Int8AttentionV2Options Stochastic = new() { Probabilities = Int8ProbabilityFormat.U8Stochastic, RoundingSeed = 3 };
    private static readonly Int8AttentionV2Options TileV = new() { ValueScalesPerTile = true };

    private (double RelL2, double Cos, double MaxAbs) Measure(string what, float[] actual, float[] expected)
    {
        (double rel, double cos, double max) m = Int8V2Measure.Of(actual, expected);
        output.WriteLine($"{what}: relL2 {m.rel:E2}, cos {m.cos:F6}, max |Δ| {m.max:E2}");
        return m;
    }

    private void MeetsBar(string what, float[] actual, float[] expected)
    {
        (double rel, double cos, _) = Measure(what, actual, expected);
        Assert.True(rel <= Int8V2Measure.BarRelL2 && cos >= Int8V2Measure.BarCos,
            $"{what}: relL2 {rel:E2} / cos {cos:F6} misses CK's bar ({Int8V2Measure.BarRelL2:E0} / {Int8V2Measure.BarCos}).");
    }

    // ── The arithmetic the kernel reproduces bit for bit ──

    [Fact]
    public void KeyMean_SumsChunksOf64InKeyOrderThenDivides()
    {
        // In F32 a running sum loses each +1 next to ±1e8; only the chunk-then-chunks order lands exactly on 0.
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = "mean-order", Sq = 1, Skv = 128, HeadDim = 64, DType = DType.F32, Fill = Int8V2Fill.Uniform,
            Mutate = (_, k, _) =>
            {
                for (int j = 0; j < 128; j++) k[j * 64] = j == 0 ? 1e8f : j == 64 ? -1e8f : 1f;
                for (int j = 0; j < 128; j++) k[j * 64 + 1] = 1f;
            },
        });
        Int8AttentionV2Operands ops = Int8AttentionV2Reference.Quantize(c.Call, Base);
        Assert.Equal(0f, ops.KeyMeans[0]);
        Assert.Equal(1f, ops.KeyMeans[1]);
    }

    [Fact]
    public void Bytes_RoundHalfToEven_ScalesAreAbsmaxOver127_AndZeroRowsGetScaleOne()
    {
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = "rounding", Sq = 2, Skv = 8, HeadDim = 40, DType = DType.F32, Fill = Int8V2Fill.Uniform,
            Mutate = (q, _, v) =>
            {
                Array.Clear(q);
                (q[0], q[1], q[2], q[3]) = (127f, 63.5f, 62.5f, -0.5f);
                for (int j = 0; j < 8; j++) v[j * 40 + 3] = j * 254f / 7f;
            },
        });
        Int8AttentionV2Operands ops = Int8AttentionV2Reference.Quantize(c.Call, Base);
        Assert.Equal(64, ops.PaddedHeadDim);
        Assert.Equal(1f, ops.QueryScales[0]);
        Assert.Equal(new[] { 127, 64, 62, 0 }, ops.Queries.AsSpan(0, 4).ToArray().Select(b => (int)b));
        Assert.Equal(1f, ops.QueryScales[1]);
        Assert.All(ops.Queries.AsSpan(64, 64).ToArray(), b => Assert.Equal(0, b));
        Assert.Equal(254f / 127f, ops.ValueScales[3]);
        for (int j = 0; j < 8; j++)
            Assert.All(ops.Keys.AsSpan(j * 64 + 40, 24).ToArray().Concat(ops.Values.AsSpan(j * 64 + 40, 24).ToArray()), b => Assert.Equal(0, b));
    }

    [Theory]
    [InlineData(Int8KeyScaleGrouping.ThreadClass, 64)]
    [InlineData(Int8KeyScaleGrouping.ThreadClass, 128)]
    [InlineData(Int8KeyScaleGrouping.PerKey, 128)]
    public void KeyScales_GroupByLaneClassWithinATile(Int8KeyScaleGrouping grouping, int blockCols)
    {
        // Key 133 is tile 1 (or 2 at BC 64), class (133 mod 8) / 2 = 2: an outlier there may move that one scale only.
        Int8V2Spec spec = new() { Id = "classes", Sq = 4, Skv = 300, HeadDim = 64, Fill = Int8V2Fill.Uniform, SampledRows = 0 };
        Int8AttentionV2Options options = new() { KeyScales = grouping, BlockCols = blockCols, SmoothKeys = false };
        using Int8V2Case plain = Int8V2Case.Create(spec);
        using Int8V2Case outlier = Int8V2Case.Create(spec with { Mutate = (_, k, _) => k[133 * 64 + 9] = 50f });
        float[] a = Int8AttentionV2Reference.Quantize(plain.Call, options).KeyScales;
        float[] b = Int8AttentionV2Reference.Quantize(outlier.Call, options).KeyScales;
        int[] changed = Enumerable.Range(0, a.Length).Where(i => a[i] != b[i]).ToArray();
        int expected = grouping == Int8KeyScaleGrouping.PerKey ? 133 : 133 / blockCols * 4 + (133 & 7) / 2;
        Assert.Equal(new[] { expected }, changed);
    }

    [Fact]
    public void SuppliedScales_RoundTripBitwise_AndAreTheOnesUsed()
    {
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec { Id = "supplied", Heads = 4, KvHeads = 2, Sq = 40, Skv = 300, HeadDim = 64 });
        Int8AttentionV2Options options = TileV;
        Int8AttentionV2Operands own = Int8AttentionV2Reference.Quantize(c.Call, options, c.Rows);
        Int8AttentionV2Operands again = Int8AttentionV2Reference.Quantize(c.Call, options, c.Rows, scales: own);
        Assert.Equal(own.Queries, again.Queries);
        Assert.Equal(own.Keys, again.Keys);
        Assert.Equal(own.Values, again.Values);
        Assert.Equal(own.QueryMeanDots, again.QueryMeanDots);
        float[] reference = c.Attend(own);

        // F16 inputs put values on exact rounding ties, so a scale one ulp off can flip a byte and move the output by one
        // quantum: a kernel is compared with the emulator running on the kernel's own scales, never across two sets of scales.
        own.KeyScales[5] = MathF.BitIncrement(own.KeyScales[5]);
        own.ValueScales[7] = MathF.BitIncrement(own.ValueScales[7]);
        Int8AttentionV2Operands nudged = Int8AttentionV2Reference.Quantize(c.Call, options, c.Rows, scales: own);
        Assert.Equal(own.KeyScales[5], nudged.KeyScales[5]);
        Assert.Equal(own.ValueScales[7], nudged.ValueScales[7]);
        Assert.True(Measure("one-ulp scales", c.Attend(nudged), reference).RelL2 < 1e-3);

        // A supplied scale is the one the bytes are quantized with: doubling it halves every byte of its class (tile 1, class 1).
        Int8AttentionV2Operands doubled = Int8AttentionV2Reference.Quantize(c.Call, options, c.Rows, scales: own);
        doubled.KeyScales[5] *= 2f;
        Int8AttentionV2Operands requantized = Int8AttentionV2Reference.Quantize(c.Call, options, c.Rows, scales: doubled);
        int[] keys = Enumerable.Range(128, 128).Where(j => (j & 7) >> 1 == 1).ToArray();
        Assert.All(keys.SelectMany(j => Enumerable.Range(j * 64, 64)),
            i => Assert.True(Math.Abs(requantized.Keys[i] - nudged.Keys[i] / 2.0) <= 1.0, $"byte {i}: {nudged.Keys[i]} → {requantized.Keys[i]}"));
        Assert.Contains(keys.SelectMany(j => Enumerable.Range(j * 64, 64)), i => requantized.Keys[i] != nudged.Keys[i]);

        // The stats pass makes every scale NaN, +∞ or at least the smallest normal float; a supplied one must be too.
        Int8AttentionV2Operands Supplied(Action<Int8AttentionV2Operands> edit)
        {
            Int8AttentionV2Operands scales = Int8AttentionV2Reference.Quantize(c.Call, options, c.Rows, scales: own);
            edit(scales);
            return Int8AttentionV2Reference.Quantize(c.Call, options, c.Rows, scales: scales);
        }
        Assert.Throws<ArgumentException>(() => Supplied(s => s.QueryScales[0] = 0f));
        Assert.Throws<ArgumentException>(() => Supplied(s => s.KeyScales[5] = -1f));
        Assert.Contains("V scale 7 ",
            Assert.Throws<ArgumentException>(() => Supplied(s => s.ValueScales[7] = float.Epsilon)).Message);
        Assert.True(float.IsNaN(Supplied(s => s.KeyScales[5] = float.NaN).KeyScales[5]));
        Assert.Equal(float.PositiveInfinity, Supplied(s => s.ValueScales[7] = float.PositiveInfinity).ValueScales[7]);
    }

    [Fact]
    public void Int32Accumulators_ReachTheirStatedBoundsExactly()
    {
        // Every Q and K byte +127 and every P byte 255 times every V byte +127: both INT32 sums sit on their bounds.
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = "bounds", Sq = 2, Skv = 256, HeadDim = 256, DType = DType.F32, SampledRows = 0,
            Mutate = (q, k, v) => { Array.Fill(q, 1f); Array.Fill(k, 1f); Array.Fill(v, 3f); },
        });
        Int8AttentionV2Options options = new() { SmoothKeys = false };
        Int8AttentionV2Operands ops = Int8AttentionV2Reference.Quantize(c.Call, options);
        Assert.Equal(Int8AttentionV2Reference.QkDotBound(256), Int8AttentionV2Reference.QkDot(ops, 0, 0, 0, 255));
        Assert.Equal(4_129_024, Int8AttentionV2Reference.QkDotBound(256));
        Assert.Equal(4_145_280, Int8AttentionV2Reference.PvPartialBound(128));
        Assert.True(Int8AttentionV2Reference.PvPartialBound(128) < 1 << 24);
        float[] o = c.Emulate(options);
        Assert.All(o, x => Assert.Equal(3f, x, 5));
    }

    [Fact]
    public void E4M3_RoundsToNearestEvenAndSaturates()
    {
        (float In, float Out)[] cases =
        [
            (448f, 448f), (463.9f, 448f), (1000f, 448f), (float.NegativeInfinity, -448f), (240f, 240f), (0.75f, 0.75f),
            (448f / 600f, 0.75f), (1.0625f, 1f), (1.1875f, 1.25f), (0.0009765625f, 0f), (0.00146484375f, 0.001953125f), (-3.3f, -3.25f),
        ];
        foreach ((float x, float y) in cases) Assert.Equal(y, Int8AttentionV2Reference.RoundToE4M3(x));
        Assert.True(float.IsNaN(Int8AttentionV2Reference.RoundToE4M3(float.NaN)));
    }

    // ── Agreement with the contract oracle ──

    [Fact]
    public void PhaseZeroMethod_MeetsCksBar_AndFp8PvDoesNot()
    {
        // Uniform [-1, 1) F16 inputs at ck-parity-4k, three heads and 64 rows each: the method CK's 6e-3 was measured with.
        Int8V2Spec[] heads = Int8V2Shapes.HeadsOf(Int8V2Shapes.Phase0.Single(s => s.Id == "ck-parity-4k"), Int8V2Fill.Uniform).ToArray();
        List<float> expected = [];
        Dictionary<string, List<float>> actual = new() { ["base"] = [], ["tile-v"] = [], ["f16"] = [], ["fp8"] = [] };
        foreach (Int8V2Spec spec in heads)
        {
            using Int8V2Case c = Int8V2Case.Create(spec);
            expected.AddRange(c.Oracle());
            actual["base"].AddRange(c.Emulate(Base));
            actual["tile-v"].AddRange(c.Emulate(TileV));
            actual["f16"].AddRange(c.Emulate(F16Pv));
            actual["fp8"].AddRange(c.Emulate(Fp8Pv));
        }
        MeetsBar("base", [.. actual["base"]], [.. expected]);
        MeetsBar("per-tile V", [.. actual["tile-v"]], [.. expected]);
        MeetsBar("f16 P×V", [.. actual["f16"]], [.. expected]);
        Assert.True(Measure("fp8 P×V", [.. actual["fp8"]], [.. expected]).RelL2 > 2e-2, "E4M3 V's three mantissa bits should cost more than s8.");
    }

    [Fact]
    public void GaussianInputs_StayInTheirEnvelope_WhichInt8QkAloneKeepsAboveTheBar()
    {
        // N(0,1) Q and K make unit-variance scores; INT8 QK's score noise (~0.011 nats) then exceeds CK's bar whatever P and V use.
        (string, int, int, int, int, bool) flux = Int8V2Shapes.Phase0.Single(s => s.Id == "flux-1024");
        using Int8V2Case c = Int8V2Case.Create(Int8V2Shapes.HeadsOf(flux, Int8V2Fill.Gaussian).First());
        float[] expected = c.Oracle();
        (double rel, double cos, _) = Measure("base", c.Emulate(Base), expected);
        Assert.True(rel <= 2e-2 && cos >= 0.9998, $"base relL2 {rel:E2} cos {cos:F6}");
        (double f16, _, _) = Measure("f16 P×V", c.Emulate(F16Pv), expected);
        Assert.True(f16 is > 6e-3 and <= 1.4e-2, $"f16 P×V relL2 {f16:E2}: INT8 QK's floor on unit-variance scores has moved.");
    }

    public static IEnumerable<object[]> ShapeRows() =>
    [
        ["b2-h3-tails", 2, 3, 3, 70, 200, 64],
        ["g1", 1, 4, 4, 33, 160, 64],
        ["g2", 1, 4, 2, 33, 160, 64],
        ["g4", 1, 8, 2, 33, 160, 128],
        ["g7", 1, 7, 1, 33, 160, 64],
        ["g8", 1, 8, 1, 33, 160, 64],
        ["d40", 1, 2, 2, 65, 300, 40],
        ["d96", 1, 2, 1, 65, 300, 96],
        ["d128", 1, 2, 2, 65, 300, 128],
        ["d256", 1, 2, 2, 65, 300, 256],
        ["sq1", 1, 4, 2, 1, 777, 128],
    ];

    [Theory]
    [MemberData(nameof(ShapeRows))]
    public void Shapes_MeetTheBarAgainstTheOracle(string id, int batch, int heads, int kvHeads, int sq, int skv, int d)
    {
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = id, Batch = batch, Heads = heads, KvHeads = kvHeads, Sq = sq, Skv = skv, HeadDim = d, Fill = Int8V2Fill.Uniform, SampledRows = 0,
        });
        float[] expected = c.Oracle();
        MeetsBar($"{id} base", c.Emulate(Base), expected);
        MeetsBar($"{id} per-tile V", c.Emulate(TileV), expected);
    }

    public static IEnumerable<object[]> MaskRows() =>
    [
        ["causal-square"], ["causal-chunk"], ["window-left"], ["window-both"], ["block-causal"], ["bias-runs"], ["kvlen-tail"], ["everything"],
    ];

    private static Int8V2Spec MaskSpec(string id)
    {
        float[] bias = new float[400];
        for (int j = 0; j < 400; j++)
            bias[j] = j is >= 120 and < 140 or >= 250 and < 390 ? float.NegativeInfinity : 0.3f * MathF.Sin(j * 0.1f);
        Int8V2Spec spec = new() { Id = id, Heads = 4, KvHeads = 2, Sq = 300, Skv = 400, HeadDim = 64, Fill = Int8V2Fill.Uniform, SampledRows = 0 };
        return id switch
        {
            "causal-square" => spec with { Skv = 300, Causal = true },
            "causal-chunk" => spec with { Sq = 70, KvLen = 333, QueryOffset = 263, Causal = true },
            "window-left" => spec with { Skv = 300, WindowLeft = 77 },
            "window-both" => spec with { Skv = 300, WindowLeft = 50, WindowRight = 19 },
            "block-causal" => spec with { Skv = 300, Causal = true, BlockSize = 24 },
            "bias-runs" => spec with { KeyBias = bias },
            "kvlen-tail" => spec with { KvLen = 333 },
            _ => spec with { Sq = 100, KvLen = 391, QueryOffset = 291, Causal = true, WindowLeft = 150, KeyBias = bias },
        };
    }

    [Theory]
    [MemberData(nameof(MaskRows))]
    public void Masks_MeetTheBarAgainstTheOracle_AndTileHeightNeverMovesARow(string id)
    {
        using Int8V2Case c = Int8V2Case.Create(MaskSpec(id));
        float[] expected = c.Oracle();
        float[] rows64 = c.Emulate(Base);
        MeetsBar($"{id} base", rows64, expected);
        // Full tiles skip the element mask, so a misclassified tile would move a row between the two tile heights.
        Assert.Equal(rows64, c.Emulate(Base with { BlockRows = 128 }));
        MeetsBar($"{id} BC64", c.Emulate(Base with { BlockCols = 64 }), expected);
    }

    [Fact]
    public void ClassifyTile_SkipsFullsAndPartials()
    {
        using Int8V2Case causal = Int8V2Case.Create(MaskSpec("causal-square"));
        AttentionCall call = causal.Call;
        Assert.Equal(AttentionTileClass.Partial, Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 0, 64, 0, 128));
        Assert.Equal(AttentionTileClass.Full, Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 128, 64, 0, 128));
        Assert.Equal(AttentionTileClass.Skip, Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 0, 64, 128, 128));
        Assert.Equal(AttentionTileClass.Partial, Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 256, 64, 256, 128));
        // Tile ends past int.MaxValue still classify; negative bounds are refused.
        Assert.Equal(AttentionTileClass.Partial,
            Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 256, 64, 128, int.MaxValue));
        Assert.Equal(AttentionTileClass.Full,
            Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 256, int.MaxValue, 0, 128));
        Assert.Throws<ArgumentException>(() => Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 0, 64, -64, 128));
        using Int8V2Case biased = Int8V2Case.Create(MaskSpec("bias-runs"));
        Assert.Equal(AttentionTileClass.Skip, Int8AttentionV2Reference.ClassifyTile(biased.Call, 0, 1, 0, 64, 256, 128));
        Assert.Equal(AttentionTileClass.Partial, Int8AttentionV2Reference.ClassifyTile(biased.Call, 0, 1, 0, 64, 64, 64));
        Assert.Equal(AttentionTileClass.Full, Int8AttentionV2Reference.ClassifyTile(biased.Call, 0, 1, 0, 64, 0, 64));
    }

    [Fact]
    public void Layouts_TokenMajorAndFusedQkv_AreBitwiseTheHeadMajorResult()
    {
        Int8V2Spec spec = new() { Id = "layouts", Batch = 2, Heads = 6, KvHeads = 2, Sq = 150, Skv = 150, HeadDim = 64, SampledRows = 0 };
        using Int8V2Case hm = Int8V2Case.Create(spec);
        using Int8V2Case tm = Int8V2Case.FromArrays(spec with { Layout = Int8V2Layout.TokenMajor }, hm.Q, hm.K, hm.V);
        using Int8V2Case fused = Int8V2Case.FromArrays(spec with { Layout = Int8V2Layout.FusedQkv }, hm.Q, hm.K, hm.V);
        float[] expected = hm.Emulate(Base);
        Assert.Equal(expected, tm.Emulate(Base));
        Assert.Equal(expected, fused.Emulate(Base));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(7)]
    public void Gqa_IsBitwiseTheCallWithKvRepeated(int group)
    {
        Int8V2Spec spec = new() { Id = $"gqa{group}", Heads = group * 2, KvHeads = 2, Sq = 50, Skv = 200, HeadDim = 64, SampledRows = 0 };
        using Int8V2Case gqa = Int8V2Case.Create(spec);
        int n = spec.Skv * spec.HeadDim;
        float[] k = new float[spec.Heads * n], v = new float[k.Length];
        for (int h = 0; h < spec.Heads; h++)
        {
            gqa.K.AsSpan(h / group * n, n).CopyTo(k.AsSpan(h * n));
            gqa.V.AsSpan(h / group * n, n).CopyTo(v.AsSpan(h * n));
        }
        using Int8V2Case mha = Int8V2Case.FromArrays(spec with { KvHeads = spec.Heads }, gqa.Q, k, v);
        Assert.Equal(mha.Emulate(Stochastic), gqa.Emulate(Stochastic));
    }

    [Fact]
    public void HeadDim40_PadsTo64_WithTheModelsScale()
    {
        Int8V2Spec spec = new() { Id = "d40-scale", Heads = 2, Sq = 64, Skv = 256, HeadDim = 40, Fill = Int8V2Fill.Uniform, SampledRows = 0 };
        using Int8V2Case c = Int8V2Case.Create(spec);
        Assert.Equal(1f / MathF.Sqrt(40f), c.Call.Scale);
        float[] expected = c.Oracle();
        double right = Measure("d40 at 1/√40", c.Emulate(Base), expected).RelL2;
        using Int8V2Case wrong = Int8V2Case.FromArrays(spec with { Scale = 1f / 8f }, c.Q, c.K, c.V);
        Assert.True(Measure("d40 at 1/√64", wrong.Emulate(Base), expected).RelL2 > 5 * right);
    }

    // ── Adversarial cases: what each variant does, asserted as it is ──

    [Theory]
    [InlineData(0, 128)]
    [InlineData(127, 128)]
    [InlineData(128, 128)]
    [InlineData(4100, 128)]
    [InlineData(8192, 128)]
    [InlineData(0, 64)]
    [InlineData(4100, 64)]
    public void SubThresholdMass_SpreadOverTiles_LosesOnlyTheTailSharingTheDominantsTile(int dominantAt, int blockCols)
    {
        using Int8V2Case c = Int8V2Case.Create(Int8V2Adversarial.SubThresholdSpread(dominantAt));
        float exact = c.Oracle()[0];
        int tile0 = dominantAt / blockCols * blockCols, lost = Math.Min(tile0 + blockCols, 8193) - tile0 - 1;
        double kept = (8192.0 - lost) / 600.0;
        float u8 = c.Emulate(Base with { BlockCols = blockCols })[0];
        output.WriteLine($"exact {exact:F6}, u8 {u8:F6}, predicted {kept / (1 + kept):F6} ({lost} tail keys share the dominant's tile)");
        Assert.Equal(8192.0 / 600 / (1 + 8192.0 / 600), exact, 5);
        Assert.Equal(kept / (1 + kept), u8, 4);
    }

    [Fact]
    public void SubThresholdMass_InOneTile_IsLostByU8_AndTheMitigationsRecoverIt()
    {
        // 127 tail keys at 1/600 of the tile's maximum round to byte 0: exact attention returns 0.1747, u8 returns 0. At most
        // (BC − 1)/510 of a tile's maximum can be lost this way, i.e. under 20% of a row at BC = 128.
        using Int8V2Case c = Int8V2Case.Create(Int8V2Adversarial.SubThresholdOneTile(128));
        float exact = c.Oracle()[0];
        Assert.Equal(127.0 / 600 / (1 + 127.0 / 600), exact, 5);
        Assert.All(c.Emulate(Base), x => Assert.Equal(0f, x));

        // Stochastic rounding keeps the tail in expectation: 256 rows hash independently, so their mean sits on the exact value.
        float[] sr = c.Emulate(Stochastic);
        double[] rows = Enumerable.Range(0, c.Rows.Length).Select(r => (double)sr[r * 64]).ToArray();
        double mean = rows.Average(), sd = Math.Sqrt(rows.Select(x => (x - mean) * (x - mean)).Sum() / (rows.Length - 1));
        output.WriteLine($"stochastic: mean {mean:F5} ± {sd / Math.Sqrt(rows.Length):F5} over {rows.Length} rows, exact {exact:F5}");
        Assert.True(Math.Abs(mean - exact) < 4 * sd / Math.Sqrt(rows.Length), "stochastic rounding should be unbiased");
        Assert.True(Math.Abs(Int8V2Measure.Of(c.Emulate(Fp8Pv), c.Oracle()).RelL2) < 5e-3);
        Assert.True(Math.Abs(Int8V2Measure.Of(c.Emulate(F16Pv), c.Oracle()).RelL2) < 1e-3);
    }

    [Fact]
    public void FiveThousandNear1e4_U8KeepsTheTailCoarsely_F16ExactlyAndStochasticWithoutBias()
    {
        using Int8V2Case c = Int8V2Case.Create(Int8V2Adversarial.FiveThousandNear1e4());
        float[] expected = c.Oracle();
        double u8 = Measure("u8", c.Emulate(Base), expected).RelL2;
        double sr = Measure("u8 stochastic", c.Emulate(Stochastic), expected).RelL2;
        Assert.True(u8 <= 2e-2, $"u8 {u8:E2}");
        Assert.True(sr < u8, "removing the rounding bias of the 2-byte tail keys should help");
        Assert.True(Measure("f16 P×V", c.Emulate(F16Pv), expected).RelL2 < 1e-4);
    }

    [Fact]
    public void KChannelOutliers_NeedSmoothing()
    {
        using Int8V2Case c = Int8V2Case.Create(Int8V2Adversarial.All().Single(s => s.Id == "k-channel-outliers"));
        float[] expected = c.Oracle();
        double smoothed = Measure("smoothed", c.Emulate(Base), expected).RelL2;
        double raw = Measure("not smoothed", c.Emulate(Base with { SmoothKeys = false }), expected).RelL2;
        Assert.True(smoothed <= 2e-2 && raw > 2.5 * smoothed, $"smoothed {smoothed:E2}, not smoothed {raw:E2}");
    }

    [Fact]
    public void VTokenOutliers_SwampASequenceWideVScale_PerTileScalesContainThem()
    {
        using Int8V2Case c = Int8V2Case.Create(Int8V2Adversarial.All().Single(s => s.Id == "v-token-outliers"));
        float[] expected = c.Oracle();
        double sequence = Measure("sequence-wide V scale", c.Emulate(Base), expected).RelL2;
        double tile = Measure("per-tile V scale", c.Emulate(TileV), expected).RelL2;
        Assert.True(sequence > 5e-2 && tile < sequence / 2.5, $"sequence {sequence:E2}, per tile {tile:E2}");
        Assert.True(Measure("f16 P×V", c.Emulate(F16Pv), expected).RelL2 < 1.5e-2);
    }

    [Fact]
    public void ValuesAboveF16Max_StayAccurateOnS8V_AndOverflowF16V()
    {
        using Int8V2Case c = Int8V2Case.Create(Int8V2Adversarial.All().Single(s => s.Id == "v-above-f16-max"));
        float[] expected = c.Oracle();
        Assert.Contains(c.V, x => MathF.Abs(x) > 65504f);
        Assert.True(Measure("s8 V", c.Emulate(Base), expected).RelL2 <= 2e-2);
        Assert.Contains(c.Emulate(F16Pv), x => !float.IsFinite(x));
    }

    [Fact]
    public void UniformAttention_AndASingleDominantKey_CarryOnlyTheVQuantizationNoise()
    {
        using Int8V2Case uniform = Int8V2Case.Create(Int8V2Adversarial.All().Single(s => s.Id == "uniform-attention-16k"));
        float[] expected = uniform.Oracle();
        Assert.True(Measure("uniform u8", uniform.Emulate(Base), expected).RelL2 <= 1.5e-2);
        Assert.True(Measure("uniform f16 P×V", uniform.Emulate(F16Pv), expected).RelL2 < 1e-6);
        using Int8V2Case dominant = Int8V2Case.Create(Int8V2Adversarial.SingleDominantKey());
        expected = dominant.Oracle();
        Assert.True(Measure("dominant u8", dominant.Emulate(Base), expected).RelL2 <= 1e-2);
        Assert.True(Measure("dominant f16 P×V", dominant.Emulate(F16Pv), expected).RelL2 < 3e-4);
    }

    // ── Smoothing's corrections: softcap, sinks, LSE, segments ──

    private static Int8V2Spec Shifted(string id, int[] channels, float offset) => new()
    {
        Id = id, Heads = 2, Sq = 64, Skv = 300, HeadDim = 64, Fill = Int8V2Fill.Uniform, SampledRows = 0,
        Mutate = (_, k, _) => { for (int j = 0; j < 300; j++) foreach (int ch in channels) k[j * 64 + ch] += offset; },
    };

    /// <summary>The same case with every key smoothed by its exact channel mean and nothing corrected: a kernel that forgot.</summary>
    private static Int8V2Case Uncorrected(Int8V2Case c, int[] segmentStarts)
    {
        Int8V2Spec s = c.Spec;
        float[] k = (float[])c.K.Clone();
        int[] bounds = [0, .. segmentStarts, s.Skv];
        for (int seg = 0; seg + 1 < bounds.Length; seg++)
            for (int ch = 0; ch < s.HeadDim; ch++)
            {
                double mean = 0;
                for (int j = bounds[seg]; j < bounds[seg + 1]; j++) mean += k[j * s.HeadDim + ch];
                mean /= bounds[seg + 1] - bounds[seg];
                for (int j = bounds[seg]; j < bounds[seg + 1]; j++) k[j * s.HeadDim + ch] -= (float)mean;
            }
        return Int8V2Case.FromArrays(s with { DType = DType.F32 }, c.Q, k, c.V);
    }

    [Fact]
    public void Softcap_UnderSmoothing_RestoresTheShiftBeforeTheTanh()
    {
        using Int8V2Case c = Int8V2Case.Create(Shifted("softcap", [2, 9, 30], 10f) with { Softcap = 1f });
        float[] expected = c.Oracle();
        MeetsBar("smoothed + softcap", c.Emulate(Base), expected);
        double raw = Measure("not smoothed + softcap", c.Emulate(Base with { SmoothKeys = false }), expected).RelL2;
        Assert.True(raw > Int8V2Measure.BarRelL2, "a large common K component should cost an unsmoothed K its range");
        using Int8V2Case forgot = Uncorrected(c, []);
        Assert.True(Measure("softcap without the restore", forgot.Oracle(), expected).RelL2 > 10 * Int8V2Measure.BarRelL2);
    }

    [Fact]
    public void SinkAndLse_UnderSmoothing_TakeTheRowShift()
    {
        Int8V2Spec spec = Shifted("sink-lse", [4, 17], 10f) with { Sinks = [6f, 5f] };
        using Int8V2Case c = Int8V2Case.Create(spec);
        float[] lseRef = new float[2 * 64], lse = new float[2 * 64], lseForgot = new float[2 * 64];
        float[] expected = c.Oracle(lseRef);
        MeetsBar("smoothed + sink", c.Emulate(Base, lse), expected);
        double lseErr = lse.Zip(lseRef, (a, b) => Math.Abs(a - b)).Max();
        output.WriteLine($"LSE max |Δ| {lseErr:E2}");
        Assert.True(lseErr < 2e-3, $"LSE off by {lseErr:E2}");
        using Int8V2Case forgot = Uncorrected(c, []);
        Assert.True(Measure("sink not shifted", forgot.Oracle(lseForgot), expected).RelL2 > 10 * Int8V2Measure.BarRelL2);
        Assert.True(lseForgot.Zip(lseRef, (a, b) => Math.Abs(a - b)).Max() > 0.5, "the row shift should be visible in an uncorrected LSE");
    }

    [Fact]
    public void Segments_SmoothedWithTheirOwnMeans_TakeTheirOwnShifts()
    {
        // Keys 0-119 carry +20 on channels 3 and 11, keys 120-299 carry −10 on channel 40: one mean leaves both segments offset.
        Int8V2Spec spec = new()
        {
            Id = "segments", Heads = 2, Sq = 64, Skv = 300, HeadDim = 64, Fill = Int8V2Fill.Uniform, SampledRows = 0,
            Mutate = (_, k, _) =>
            {
                for (int j = 0; j < 300; j++)
                    if (j < 120) { k[j * 64 + 3] += 20f; k[j * 64 + 11] += 20f; }
                    else k[j * 64 + 40] -= 10f;
            },
        };
        using Int8V2Case c = Int8V2Case.Create(spec);
        float[] expected = c.Oracle();
        Int8AttentionV2Options segmented = Base with { KeySegmentStarts = [120] };
        MeetsBar("two segments", c.Emulate(segmented), expected);
        Assert.True(Measure("one mean", c.Emulate(Base), expected).RelL2 > Measure("two means", c.Emulate(segmented), expected).RelL2,
            "a mean per segment should leave K less range to cover than one mean for both");
        using Int8V2Case capped = Int8V2Case.FromArrays(spec with { Softcap = 3f }, c.Q, c.K, c.V);
        MeetsBar("two segments + softcap", capped.Emulate(segmented), capped.Oracle());
        using Int8V2Case forgot = Uncorrected(c, [120]);
        Assert.True(Measure("segments without their shifts", forgot.Oracle(), expected).RelL2 > 10 * Int8V2Measure.BarRelL2);
    }

    // ── Exclusions, NaN, determinism, refusals ──

    [Fact]
    public void FullyExcludedRows_WriteZeros_LseMinusInfinity_OrTheSinkAlone()
    {
        float[] bias = Enumerable.Repeat(float.NegativeInfinity, 200).ToArray();
        Int8V2Spec spec = new() { Id = "excluded", Heads = 2, Sq = 8, Skv = 200, HeadDim = 64, KeyBias = bias, SampledRows = 0 };
        using Int8V2Case c = Int8V2Case.Create(spec);
        float[] lse = new float[16];
        Assert.All(c.Emulate(Base, lse), x => Assert.Equal(0f, x));
        Assert.All(lse, x => Assert.Equal(float.NegativeInfinity, x));
        using Int8V2Case sinks = Int8V2Case.FromArrays(spec with { Sinks = [0.25f, -2f] }, c.Q, c.K, c.V);
        Assert.All(sinks.Emulate(Base, lse), x => Assert.Equal(0f, x));
        Assert.Equal(0.25f, lse[0], 5);
        Assert.Equal(-2f, lse[8], 5);

        // A −∞ sink is no sink at all: an excluded row stays zero with LSE −∞, a visible one is untouched bit for bit.
        using Int8V2Case none = Int8V2Case.FromArrays(spec with { Sinks = [float.NegativeInfinity, -2f] }, c.Q, c.K, c.V);
        Assert.All(none.Emulate(Base, lse), x => Assert.Equal(0f, x));
        Assert.Equal(float.NegativeInfinity, lse[0]);
        using Int8V2Case open = Int8V2Case.FromArrays(spec with { KeyBias = null }, c.Q, c.K, c.V);
        using Int8V2Case openNone = Int8V2Case.FromArrays(spec with { KeyBias = null, Sinks = [float.NegativeInfinity, float.NegativeInfinity] },
            c.Q, c.K, c.V);
        Assert.Equal(open.Emulate(Base), openNone.Emulate(Base));
    }

    [Fact]
    public void NaN_IsPropagatedNeverZeroed()
    {
        Int8V2Spec spec = new() { Id = "nan", Heads = 2, KvHeads = 2, Sq = 16, Skv = 200, HeadDim = 64, DType = DType.F32, SampledRows = 0 };
        using Int8V2Case clean = Int8V2Case.Create(spec);
        float[] lse = new float[32];

        float[] q = (float[])clean.Q.Clone();
        q[5 * 64 + 7] = float.NaN;
        using (Int8V2Case c = Int8V2Case.FromArrays(spec, q, clean.K, clean.V))
        {
            float[] o = c.Emulate(Base, lse);
            for (int r = 0; r < 16; r++) Assert.Equal(r == 5, float.IsNaN(o[r * 64]));
            Assert.True(float.IsNaN(lse[5]) && float.IsFinite(lse[4]));
        }

        float[] k = (float[])clean.K.Clone();
        k[200 * 64 + 150 * 64 + 1] = float.NaN;
        using (Int8V2Case c = Int8V2Case.FromArrays(spec, clean.Q, k, clean.V))
        {
            float[] o = c.Emulate(Base);
            Assert.All(o.AsSpan(0, 16 * 64).ToArray(), x => Assert.True(float.IsFinite(x)));
            Assert.All(o.AsSpan(16 * 64).ToArray(), x => Assert.True(float.IsNaN(x)));
        }

        float[] v = (float[])clean.V.Clone();
        v[30 * 64 + 9] = float.NaN;
        using (Int8V2Case c = Int8V2Case.FromArrays(spec, clean.Q, clean.K, v))
            foreach (Int8AttentionV2Options options in (Int8AttentionV2Options[])[Base, TileV, F16Pv, Fp8Pv])
            {
                float[] o = c.Emulate(options);
                for (int i = 0; i < 16 * 64; i++) Assert.Equal(i % 64 == 9, float.IsNaN(o[i]));
            }

        // An infinity cannot be quantized: it reaches a scale (0·∞ in the dequant) and comes out as NaN, never as a finite value.
        float[] qInf = (float[])clean.Q.Clone();
        qInf[3 * 64 + 11] = float.PositiveInfinity;
        using (Int8V2Case c = Int8V2Case.FromArrays(spec, qInf, clean.K, clean.V))
        {
            float[] o = c.Emulate(Base, lse);
            for (int r = 0; r < 16; r++) Assert.Equal(r == 3, float.IsNaN(o[r * 64]));
            Assert.True(float.IsNaN(lse[3]) && float.IsFinite(lse[2]));
        }
        float[] kInf = (float[])clean.K.Clone();
        kInf[150 * 64 + 2] = float.PositiveInfinity;
        float[] vInf = (float[])clean.V.Clone();
        vInf[30 * 64 + 9] = float.NegativeInfinity;
        using (Int8V2Case c = Int8V2Case.FromArrays(spec with { Causal = true, QueryOffset = 140 }, clean.Q, kInf, vInf))
        {
            float[] o = c.Emulate(Base with { SmoothKeys = false, KeyScales = Int8KeyScaleGrouping.PerKey });
            for (int r = 0; r < 16; r++)
                for (int ch = 0; ch < 64; ch++) Assert.Equal(140 + r >= 150 || ch == 9, float.IsNaN(o[r * 64 + ch]));
        }

        float[] bias = new float[200];
        bias[77] = float.NaN;
        using (Int8V2Case c = Int8V2Case.FromArrays(spec with { KeyBias = bias, Causal = true, QueryOffset = 70 }, clean.Q, clean.K, clean.V))
        {
            float[] o = c.Emulate(Base);
            for (int r = 0; r < 16; r++) Assert.Equal(70 + r >= 77, float.IsNaN(o[r * 64]));
        }

        // A +∞ sink makes its row NaN (contract §3), whether the row sees keys or not, in the emulator and the oracle.
        float[] excluded = Enumerable.Repeat(float.NegativeInfinity, 200).ToArray();
        foreach (float[]? keyBias in (float[]?[])[null, excluded])
        {
            Int8V2Spec infSink = spec with { KeyBias = keyBias, Sinks = [float.PositiveInfinity, -2f] };
            using Int8V2Case c = Int8V2Case.FromArrays(infSink, clean.Q, clean.K, clean.V);
            float[] emulatedLse = new float[32], oracleLse = new float[32];
            float[] o = c.Emulate(Base, emulatedLse), expected = c.Oracle(oracleLse);
            bool[] head0 = [.. Enumerable.Range(0, o.Length).Select(i => i < 16 * 64)];
            bool[] head0Lse = [.. Enumerable.Range(0, 32).Select(i => i < 16)];
            Assert.Equal(head0, o.Select(float.IsNaN));
            Assert.Equal(head0, expected.Select(float.IsNaN));
            Assert.Equal(head0Lse, emulatedLse.Select(float.IsNaN));
            Assert.Equal(head0Lse, oracleLse.Select(float.IsNaN));
        }
    }

    [Fact]
    public void EveryVariant_IsBitwiseDeterministic()
    {
        using Int8V2Case c = Int8V2Case.Create(MaskSpec("everything") with { Sinks = [0.5f, 1f, -1f, 0f], Softcap = 4f });
        Int8AttentionV2Options[] variants =
        [
            Base, TileV, Stochastic, F16Pv, Fp8Pv, Base with { KeyScales = Int8KeyScaleGrouping.PerKey, BlockCols = 64, KeySegmentStarts = [100] },
        ];
        foreach (Int8AttentionV2Options options in variants)
        {
            float[] lse1 = new float[4 * 100], lse2 = new float[4 * 100];
            Assert.Equal(c.Emulate(options, lse1), c.Emulate(options, lse2));
            Assert.Equal(lse1, lse2);
        }
        Assert.NotEqual(c.Emulate(Stochastic), c.Emulate(Stochastic with { RoundingSeed = 4 }));
    }

    [Fact]
    public void ABiasThatDoesNotFit_IsRefusedBeforeAnyRead()
    {
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = "bias-shapes", Sq = 4, Skv = 200, HeadDim = 64, Fill = Int8V2Fill.Uniform, SampledRows = 0,
        });
        using Tensor rank1 = new(new TensorShape(200), DType.F32), shortKeys = new(new TensorShape(1, 1, 1, 199), DType.F32);
        using Tensor half = new(new TensorShape(1, 1, 1, 200), DType.F16);
        foreach (Tensor misfit in (Tensor[])[rank1, shortKeys])
        {
            AttentionCall call = c.Call with { Mask = AttentionMask.Additive(misfit) };
            Assert.Throws<ArgumentException>(() => Int8AttentionV2Reference.Run(call, Base));
            Assert.Throws<ArgumentException>(() => Int8AttentionV2Reference.ClassifyTile(call, 0, 0, 0, 4, 0, 128));
        }
        AttentionCall halfBias = c.Call with { Mask = AttentionMask.Additive(half) };
        Assert.Throws<AttentionUnsupportedException>(() =>
            Int8AttentionV2Reference.ClassifyTile(halfBias, 0, 0, 0, 4, 0, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Int8AttentionV2Reference.ClassifyTile(c.Call, 0, 1, 0, 4, 0, 128));

        // A bias broadcast over the keys is one element for every key; a constant leaves the softmax where it was.
        using Tensor broadcast = new(new TensorShape(1, 1, 1, 1), DType.F32);
        broadcast.AsSpan<float>()[0] = -3f;
        Int8AttentionV2Reference.Run(c.Call with { Mask = AttentionMask.Additive(broadcast) }, Base);
        MeetsBar("key-broadcast bias", c.Call.Output.Storage.AsReadOnlySpan<float>().ToArray(), c.Oracle());
    }

    [Fact]
    public void Exp2_IsInjectable_SoAKernelsOwnExp2ReplaysBitForBit()
    {
        // A stand-in for ex2.approx one ulp high. Its values for the arguments the emulator asks for (which never depend on an
        // earlier exp2) replay into the same bytes as running it directly: the two-pass recipe for comparing P with a GPU.
        static float Hardware(float x) => MathF.BitIncrement((float)Math.Pow(2.0, x));
        using Int8V2Case c = Int8V2Case.Create(MaskSpec("everything") with { Sinks = [0.5f, 1f, -1f, 0f] });
        foreach (Int8AttentionV2Options options in (Int8AttentionV2Options[])[Base, F16Pv])
        {
            float[] reference = c.Emulate(options);
            Dictionary<float, float> table = [];
            float[] recorded = c.Emulate(options with
            {
                Exp2 = x =>
                {
                    table[x] = Hardware(x);
                    return (float)Math.Pow(2.0, x);
                },
            });
            Assert.Equal(reference, recorded);
            float[] direct = c.Emulate(options with { Exp2 = Hardware });
            Assert.NotEqual(reference, direct);
            Assert.Equal(direct, c.Emulate(options with { Exp2 = x => table[x] }));
        }
    }

    [Fact]
    public void WhatV2DoesNotRun_IsRefused()
    {
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec { Id = "refusals", Sq = 4, Skv = 8, HeadDim = 64, SampledRows = 0 });
        AttentionCall call = c.Call;
        using Tensor slopes = new(new TensorShape(1), DType.F32), halfSinks = new(new TensorShape(1), DType.F16);
        using Tensor perQuery = new(new TensorShape(1, 1, 4, 8), DType.F32);
        Assert.Throws<AttentionUnsupportedException>(() => Int8AttentionV2Reference.Run(call with { Logits = new(0f, null, slopes) }, Base));
        Assert.Throws<AttentionUnsupportedException>(() =>
            Int8AttentionV2Reference.Run(call with { Logits = new(0f, halfSinks, null) }, Base));
        Assert.Throws<AttentionUnsupportedException>(() => Int8AttentionV2Reference.Run(call with { Mask = AttentionMask.Additive(perQuery) }, Base));
        Assert.Throws<AttentionUnsupportedException>(() =>
            Int8AttentionV2Reference.Run(call with { Positions = AttentionPositions.Device(1, 8, 0) }, Base));
        Assert.Throws<AttentionUnsupportedException>(() => Int8AttentionV2Reference.PaddedHeadDim(257));
        Assert.Throws<ArgumentException>(() => Int8AttentionV2Reference.Run(call, Base with { BlockRows = 32 }));
        Assert.Throws<ArgumentException>(() => Int8AttentionV2Reference.Run(call, Base with { KeySegmentStarts = [5, 5] }));
    }
}
