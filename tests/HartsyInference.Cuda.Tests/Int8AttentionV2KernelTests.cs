using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>INT8 attention v2 (int8_attn_v2.ptx) against its emulator, <see cref="Int8AttentionV2Reference"/>: the quantized bytes,
/// scales and means bit for bit, the INT32 QK products, tile maxima, P bytes and PV partials of a traced tile, the output bit for
/// bit when the emulator replays the kernel's exp2 and within relL2 1e-5 when it does not, and the rows' routing and refusals.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Int8AttentionV2KernelTests(ITestOutputHelper output)
{
    private static readonly Int8AttentionV2Pv[] U8Modes = [Int8AttentionV2Pv.U8SequenceScales, Int8AttentionV2Pv.U8TileScales];

    private static Int8AttentionV2Pv Pv(string name) => name switch
    {
        "seq" => Int8AttentionV2Pv.U8SequenceScales,
        "tile" => Int8AttentionV2Pv.U8TileScales,
        "f16" => Int8AttentionV2Pv.F16,
        _ => throw new ArgumentException(name),
    };

    private static Int8AttentionV2Options Options(Int8AttentionV2Pv pv) => pv switch
    {
        Int8AttentionV2Pv.U8TileScales => new() { ValueScalesPerTile = true },
        Int8AttentionV2Pv.F16 => new() { Probabilities = Int8ProbabilityFormat.F16 },
        _ => new(),
    };

    private bool Open(out CudaBackend cuda)
    {
        cuda = null!;
        if (!BackendGate.TryOpen("cuda", output.WriteLine, out IBackend? backend)) return false;
        cuda = (CudaBackend)backend!;
        return true;
    }

    /// <summary>The case's call with an output in <paramref name="outType"/>: the emulator's own output tensor is F32.</summary>
    private static (AttentionCall Call, Tensor Output) GpuCall(Int8V2Case c, DType outType)
    {
        Int8V2Spec s = c.Spec;
        Tensor o = new(new TensorShape(s.Batch, s.Heads, s.Sq, s.HeadDim), outType);
        return (c.Call with { Output = AttentionOperand.HeadMajor(o) }, o);
    }

    /// <summary>The checked rows of a head-major <c>[B, H, Sq, D]</c> output read back as bytes, widened to F32.</summary>
    private static float[] Rows(Int8V2Case c, byte[] bytes, DType type)
    {
        Int8V2Spec s = c.Spec;
        float[] rows = new float[s.Batch * s.Heads * c.Rows.Length * s.HeadDim];
        int i = 0;
        for (int b = 0; b < s.Batch; b++)
            for (int h = 0; h < s.Heads; h++)
                foreach (int r in c.Rows)
                    for (int ch = 0; ch < s.HeadDim; ch++)
                    {
                        long at = (((long)b * s.Heads + h) * s.Sq + r) * s.HeadDim + ch;
                        rows[i++] = type == DType.F32 ? BitConverter.ToSingle(bytes, (int)(at * 4))
                            : (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(bytes, (int)(at * 2)));
                    }
        return rows;
    }

    /// <summary>The emulator run on the kernel's exp2: the arguments it asks for, evaluated on the GPU, replayed.</summary>
    private static Int8AttentionV2Options Replayed(CudaBackend cuda, Int8V2Case c, Int8AttentionV2Options options)
    {
        HashSet<float> arguments = [];
        c.Emulate(options with { Exp2 = x => { arguments.Add(x); return (float)Math.Pow(2.0, x); } });
        float[] args = [.. arguments];
        float[] values = cuda.Int8V2Exp2ForTest(args);
        Dictionary<float, float> table = new(args.Length);
        for (int i = 0; i < args.Length; i++) table[args[i]] = values[i];
        return options with { Exp2 = x => table[x] };
    }

    private (double RelL2, double Cos, double MaxAbs) Measure(string what, float[] actual, float[] expected)
    {
        (double rel, double cos, double max) m = Int8V2Measure.Of(actual, expected);
        output.WriteLine($"{what}: relL2 {m.rel:E2}, cos {m.cos:F7}, max |Δ| {m.max:E2}");
        return m;
    }

    public static IEnumerable<object[]> QuantRows()
    {
        foreach (DType dt in new[] { DType.F16, DType.F32 })
            foreach (Int8V2Layout layout in new[] { Int8V2Layout.HeadMajor, Int8V2Layout.TokenMajor, Int8V2Layout.FusedQkv })
                foreach (string pv in new[] { "seq", "tile", "f16" })
                    yield return [dt.ToString(), layout, pv];
    }

    /// <summary>Q/K/V bytes, scales and K means are the emulator's bit for bit: scales and means compared directly, bytes against the
    /// emulator quantizing with the kernel's own scales. Two batches of three heads, query and key tails inside a tile.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(QuantRows))]
    public void Quant_IsTheEmulatorBitForBit(string dtype, Int8V2Layout layout, string pvName)
    {
        Int8AttentionV2Pv pv = Pv(pvName);
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        DType dt = dtype == "F32" ? DType.F32 : DType.F16;
        int sq = layout == Int8V2Layout.FusedQkv ? 333 : 200;
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = "quant", Batch = 2, Heads = 3, KvHeads = 3, Sq = sq, Skv = 333, HeadDim = 128, DType = dt, Layout = layout,
            SampledRows = 0, Mutate = (_, k, _) => { for (int j = 0; j < k.Length; j += 128) k[j + 5] += 6f; },
        });
        (AttentionCall call, Tensor o) = GpuCall(c, dt);
        using Tensor __ = o;
        Int8V2Readback gpu = cuda.RunInt8AttentionV2ForTest(call, pv, 64);
        Int8AttentionV2Options options = Options(pv);
        Int8AttentionV2Operands own = Int8AttentionV2Reference.Quantize(c.Call, options);
        int heads = 2 * 3, keys = 333, kvPad = (int)gpu.Layout.KvPad;
        float[] keyScales = new float[heads * keys];
        for (int bh = 0; bh < heads; bh++)
        {
            gpu.KScale.AsSpan(bh * kvPad, keys).CopyTo(keyScales.AsSpan(bh * keys));
            Assert.All(gpu.KScale.AsSpan(bh * kvPad + keys, kvPad - keys).ToArray(), s => Assert.Equal(1f, s));
        }
        Assert.Equal(own.QueryScales, gpu.QScale);
        Assert.Equal(own.KeyScales, keyScales);
        Assert.Equal(own.KeyMeans, gpu.KMean);
        if (pv != Int8AttentionV2Pv.F16) Assert.Equal(own.ValueScales, gpu.VScale);

        Int8AttentionV2Operands supplied = Int8AttentionV2Reference.Quantize(c.Call, options);
        gpu.QScale.CopyTo(supplied.QueryScales, 0);
        keyScales.CopyTo(supplied.KeyScales, 0);
        gpu.KMean.CopyTo(supplied.KeyMeans, 0);
        if (pv != Int8AttentionV2Pv.F16) gpu.VScale.CopyTo(supplied.ValueScales, 0);
        Int8AttentionV2Operands ops = Int8AttentionV2Reference.Quantize(c.Call, options, scales: supplied);
        Assert.Equal(ops.Queries, gpu.Q8);
        for (int bh = 0; bh < heads; bh++)
        {
            Assert.Equal(ops.Keys.AsSpan(bh * keys * 128, keys * 128).ToArray(), gpu.K8.AsSpan(bh * kvPad * 128, keys * 128).ToArray());
            Assert.All(gpu.K8.AsSpan((bh * kvPad + keys) * 128, (kvPad - keys) * 128).ToArray(), b => Assert.Equal(0, b));
            if (pv == Int8AttentionV2Pv.F16) continue;
            for (int ch = 0; ch < 128; ch++)
                for (int j = 0; j < kvPad; j++)
                {
                    sbyte expected = j < keys ? ops.Values[(bh * keys + j) * 128 + ch] : (sbyte)0;
                    sbyte actual = gpu.V8[(bh * 128 + ch) * kvPad + Int8AttentionV2Layout.PhysicalKey(j)];
                    if (expected != actual) Assert.Fail($"V byte (bh {bh}, ch {ch}, key {j}): emulator {expected}, kernel {actual}");
                }
        }
    }

    public static IEnumerable<object[]> TraceRows()
    {
        foreach (string pv in new[] { "seq", "tile" })
            foreach (int br in new[] { 64, 128 })
                yield return [pv, br];
    }

    /// <summary>One traced query tile: INT32 QK products against <see cref="Int8AttentionV2Reference.QkDot"/>, then on the kernel's own
    /// exp2 the tile maxima, P bytes and byte sums bit for bit, the INT32 PV partials exactly in integer arithmetic from the kernel's P and
    /// V bytes (the V permutation and mma wiring), and the whole F32 output bit for bit.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(TraceRows))]
    public void Trace_ProductsBytesPartialsAndOutput_AreTheEmulatorsOnItsExp2(string pvName, int blockRows)
    {
        Int8AttentionV2Pv pv = Pv(pvName);
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = "trace", Heads = 2, KvHeads = 2, Sq = 150, Skv = 300, HeadDim = 128, SampledRows = 0,
            Mutate = (_, _, v) => { for (int j = 0; j < 300; j += 37) for (int ch = 0; ch < 128; ch++) v[(300 + j) * 128 + ch] *= 6f; },
        });
        (AttentionCall call, Tensor o) = GpuCall(c, DType.F32);
        using Tensor __ = o;
        const int Head = 1, QueryTile = 1;
        Int8V2Readback gpu = cuda.RunInt8AttentionV2ForTest(call, pv, blockRows, (0, Head, QueryTile));
        Int8AttentionV2Options options = Options(pv);
        Int8AttentionV2Operands ops = Int8AttentionV2Reference.Quantize(c.Call, options);
        Assert.Equal(ops.Queries, gpu.Q8);
        int kvPad = (int)gpu.Layout.KvPad, tiles = gpu.Layout.Tiles, keys = 300;
        int row0 = QueryTile * blockRows, rows = Math.Min(blockRows, 150 - row0);
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < keys; j++)
                Assert.Equal(Int8AttentionV2Reference.QkDot(ops, 0, Head, row0 + r, j), gpu.Dots[r * kvPad + j]);

        Dictionary<(int Row, int Tile), (float Max, byte[] P, float[] Partial, int Sum)> traced = [];
        Int8AttentionV2Options replay = Replayed(cuda, c, options) with
        {
            TileTrace = (bh, row, tile, max, p, partial, sum) =>
            {
                if (bh == Head && row >= row0 && row < row0 + rows) traced[(row - row0, tile)] = (max, p.ToArray(), partial.ToArray(), sum);
            },
        };
        float[] expected = c.Emulate(replay);
        Assert.Equal(rows * tiles, traced.Count);
        // On the emulator's own exp2 the bytes may differ by one where an exp2 last-ulp difference crosses a rounding tie.
        Dictionary<(int Row, int Tile), byte[]> plain = [];
        c.Emulate(options with { TileTrace = (bh, row, tile, _, p, _, _) => { if (bh == Head && row >= row0 && row < row0 + rows) plain[(row - row0, tile)] = p.ToArray(); } });
        int flips = 0, bytes = 0;
        foreach (((int r, int t), byte[] p) in plain)
            for (int j = 0; j < p.Length; j++, bytes++)
            {
                int delta = Math.Abs(p[j] - gpu.Probabilities[r * kvPad + t * 128 + j]);
                Assert.True(delta <= 1, $"row {r} tile {t} key {j}: emulator {p[j]}, kernel {gpu.Probabilities[r * kvPad + t * 128 + j]}");
                flips += delta;
            }
        output.WriteLine($"P bytes on the emulator's own exp2: {flips} of {bytes} differ by one");
        Assert.True(flips < 1e-3 * bytes, $"{flips} of {bytes} P bytes differ");
        for (int r = 0; r < rows; r++)
            for (int t = 0; t < tiles; t++)
            {
                (float max, byte[] p, float[] partial, int sum) = traced[(r, t)];
                Assert.Equal(max, gpu.TileMax[r * tiles + t]);
                Assert.Equal(sum, gpu.Sums[r * tiles + t]);
                Assert.Equal(p, gpu.Probabilities.AsSpan(r * kvPad + t * 128, p.Length).ToArray());
                for (int ch = 0; ch < 128; ch++)
                {
                    long exact = 0;
                    for (int j = 0; j < 128; j++)
                        exact += gpu.Probabilities[r * kvPad + t * 128 + j]
                            * (long)gpu.V8[((long)Head * 128 + ch) * kvPad + Int8AttentionV2Layout.PhysicalKey(t * 128 + j)];
                    int kernel = gpu.Partials[(r * tiles + t) * 128 + ch];
                    Assert.Equal(exact, kernel);
                    Assert.Equal(partial[ch], kernel);
                }
            }
        Assert.Equal(expected, Rows(c, gpu.Output, DType.F32));
    }

    public static IEnumerable<object[]> OutputRows()
    {
        foreach (DType dt in new[] { DType.F16, DType.F32 })
            foreach (string pv in new[] { "seq", "tile", "f16" })
                foreach (int br in new[] { 64, 128 })
                    yield return [dt.ToString(), pv, br];
    }

    /// <summary>u8 PV: the emulator's output bit for bit (F16 output: rounded to F16) when it replays the kernel's exp2. On its own exp2
    /// the last-ulp differences become u8 rounding flips, so that comparison is bounded instead: relL2 1e-4 (F32 output) or 5e-4
    /// (F16 output, which carries its own rounding). F16 PV sums in the tensor cores' order: relL2 1e-5 (F32), 5e-4 (F16).
    /// Two batches, three heads, query and key tails.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(OutputRows))]
    public void Output_IsTheEmulators(string dtype, string pvName, int blockRows)
    {
        Int8AttentionV2Pv pv = Pv(pvName);
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        DType dt = dtype == "F32" ? DType.F32 : DType.F16;
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = "output", Batch = 2, Heads = 3, KvHeads = 3, Sq = 200, Skv = 333, HeadDim = 128, DType = dt, SampledRows = 0,
        });
        (AttentionCall call, Tensor o) = GpuCall(c, dt);
        using Tensor __ = o;
        Int8V2Readback gpu = cuda.RunInt8AttentionV2ForTest(call, pv, blockRows);
        float[] actual = Rows(c, gpu.Output, dt);
        Int8AttentionV2Options options = Options(pv);
        float[] emulated = c.Emulate(options);
        double rel = Measure("vs emulator", actual, emulated).RelL2;
        if (pv != Int8AttentionV2Pv.F16)
        {
            float[] replayed = c.Emulate(Replayed(cuda, c, options));
            if (dt == DType.F32) Assert.Equal(replayed, actual);
            else Assert.Equal(replayed.Select(x => (float)(Half)x).ToArray(), actual);
        }
        double bound = dt == DType.F16 ? 5e-4 : pv == Int8AttentionV2Pv.F16 ? 1e-5 : 1e-4;
        Assert.True(rel <= bound, $"relL2 {rel:E2} against {bound:E0}");
    }

    /// <summary>Edge geometries, bit for bit on the replayed exp2: one query row, fewer keys than a tile, exactly one tile, a
    /// key count one past a tile, and a batch of three with a query tail; both V scale modes, both tile heights.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(1, 1, 77)]
    [InlineData(1, 5, 128)]
    [InlineData(1, 70, 129)]
    [InlineData(3, 130, 260)]
    public void EdgeGeometries_AreTheEmulatorsOnItsExp2(int batch, int sq, int skv)
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        using Int8V2Case c = Int8V2Case.Create(new Int8V2Spec
        {
            Id = $"edge-{batch}-{sq}-{skv}", Batch = batch, Heads = 2, KvHeads = 2, Sq = sq, Skv = skv, HeadDim = 128, DType = DType.F32,
            SampledRows = 0,
        });
        foreach (Int8AttentionV2Pv pv in U8Modes)
            foreach (int br in new[] { 64, 128 })
            {
                (AttentionCall call, Tensor o) = GpuCall(c, DType.F32);
                using Tensor __ = o;
                float[] actual = Rows(c, cuda.RunInt8AttentionV2ForTest(call, pv, br).Output, DType.F32);
                Assert.Equal(c.Emulate(Replayed(cuda, c, Options(pv))), actual);
            }
    }

    /// <summary>Comfy Kitchen's published bar, relL2 ≤ 6e-3 and cos ≥ 0.99998 against the double-precision oracle, on uniform inputs at
    /// ck-parity-4k (three heads, 64 rows each), for every PV format.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void CksBar_HoldsOnUniformInputs()
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        Int8V2Spec[] heads = Int8V2Shapes.HeadsOf(Int8V2Shapes.Phase0.Single(s => s.Id == "ck-parity-4k"), Int8V2Fill.Uniform).ToArray();
        foreach (Int8AttentionV2Pv pv in new[] { Int8AttentionV2Pv.U8SequenceScales, Int8AttentionV2Pv.U8TileScales, Int8AttentionV2Pv.F16 })
        {
            List<float> expected = [], actual = [];
            foreach (Int8V2Spec spec in heads)
            {
                using Int8V2Case c = Int8V2Case.Create(spec);
                (AttentionCall call, Tensor o) = GpuCall(c, DType.F16);
                using Tensor __ = o;
                expected.AddRange(c.Oracle());
                actual.AddRange(Rows(c, cuda.RunInt8AttentionV2ForTest(call, pv, 64).Output, DType.F16));
            }
            (double rel, double cos, _) = Measure($"{pv} vs oracle", [.. actual], [.. expected]);
            Assert.True(rel <= Int8V2Measure.BarRelL2 && cos >= Int8V2Measure.BarCos, $"{pv}: relL2 {rel:E2}, cos {cos:F7}");
        }
    }

    public static IEnumerable<object[]> AdversarialRows() =>
    [
        ["k-channel-outliers"], ["k-token-outliers"], ["v-channel-outliers"], ["v-token-outliers"], ["single-dominant-key"],
        ["v-above-f16-max"], ["uniform-attention-16k"], ["sub-threshold@0"], ["sub-threshold@8192"], ["sub-threshold-one-tile"],
    ];

    private static Int8V2Spec Adversarial(string id) => id switch
    {
        "v-above-f16-max" => Int8V2Adversarial.ValuesAboveF16Max(128),
        "uniform-attention-16k" => Int8V2Adversarial.UniformAttention(128),
        "sub-threshold@0" => Int8V2Adversarial.SubThresholdSpread(0, headDim: 128),
        "sub-threshold@8192" => Int8V2Adversarial.SubThresholdSpread(8192, headDim: 128),
        "sub-threshold-one-tile" => Int8V2Adversarial.SubThresholdOneTile(128, headDim: 128),
        _ => Int8V2Adversarial.All().Single(s => s.Id == id),
    };

    /// <summary>The adversarial cases at head dim 128: the kernel agrees with the emulator on each (both V scale modes, and F16 PV),
    /// whatever the emulator's own error against the oracle is; that error is printed for the record.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(AdversarialRows))]
    public void Adversarial_KernelMatchesTheEmulator(string id)
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        using Int8V2Case c = Int8V2Case.Create(Adversarial(id));
        DType dt = c.Spec.DType;
        float[] oracle = c.Oracle();
        foreach (Int8AttentionV2Pv pv in new[] { Int8AttentionV2Pv.U8SequenceScales, Int8AttentionV2Pv.U8TileScales, Int8AttentionV2Pv.F16 })
        {
            (AttentionCall call, Tensor o) = GpuCall(c, dt);
            using Tensor __ = o;
            float[] actual = Rows(c, cuda.RunInt8AttentionV2ForTest(call, pv, 64).Output, dt);
            float[] emulated = c.Emulate(Options(pv));
            Measure($"{id} {pv} emulator vs oracle", emulated, oracle);
            (double rel, _, _) = Measure($"{id} {pv} kernel vs emulator", actual, emulated);
            double bound = dt == DType.F16 ? 5e-4 : pv == Int8AttentionV2Pv.F16 ? 1e-5 : 1e-4;
            Assert.True(rel <= bound || pv == Int8AttentionV2Pv.F16 && emulated.Any(x => !float.IsFinite(x)), $"{id} {pv}: relL2 {rel:E2}");
            if (pv != Int8AttentionV2Pv.F16)
            {
                float[] replayed = c.Emulate(Replayed(cuda, c, Options(pv)));
                Assert.Equal(dt == DType.F32 ? replayed : [.. replayed.Select(x => (float)(Half)x)], actual);
            }
        }
    }

    /// <summary>NaN in Q poisons its row, NaN in K its head, NaN in V its channel; never a zero.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void NaN_PropagatesAsTheEmulatorSays()
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        Int8V2Spec spec = new() { Id = "nan", Heads = 2, KvHeads = 2, Sq = 16, Skv = 200, HeadDim = 128, DType = DType.F32, SampledRows = 0 };
        using Int8V2Case clean = Int8V2Case.Create(spec);
        float[] q = (float[])clean.Q.Clone(), k = (float[])clean.K.Clone(), v = (float[])clean.V.Clone();
        q[5 * 128 + 7] = float.NaN;
        k[200 * 128 + 150 * 128 + 1] = float.NaN;
        v[30 * 128 + 9] = float.NaN;
        using Int8V2Case c = Int8V2Case.FromArrays(spec, q, k, v);
        foreach (Int8AttentionV2Pv pv in U8Modes)
        {
            (AttentionCall call, Tensor o) = GpuCall(c, DType.F32);
            using Tensor __ = o;
            float[] actual = Rows(c, cuda.RunInt8AttentionV2ForTest(call, pv, 64).Output, DType.F32);
            float[] emulated = c.Emulate(Options(pv));
            Assert.Equal(emulated.Select(float.IsNaN), actual.Select(float.IsNaN));
            Assert.Contains(actual, float.IsNaN);
            Assert.Contains(actual, float.IsFinite);
        }
    }

    /// <summary>Head-major, token-major and fused-QKV operands with the same values give the same bits, run twice.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Layouts_AreBitwiseEqual_AndRunsAreDeterministic()
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        Int8V2Spec spec = new() { Id = "layouts", Batch = 2, Heads = 4, KvHeads = 4, Sq = 300, Skv = 300, HeadDim = 128, SampledRows = 0 };
        using Int8V2Case hm = Int8V2Case.Create(spec);
        using Int8V2Case tm = Int8V2Case.FromArrays(spec with { Layout = Int8V2Layout.TokenMajor }, hm.Q, hm.K, hm.V);
        using Int8V2Case fused = Int8V2Case.FromArrays(spec with { Layout = Int8V2Layout.FusedQkv }, hm.Q, hm.K, hm.V);
        foreach (Int8AttentionV2Pv pv in new[] { Int8AttentionV2Pv.U8SequenceScales, Int8AttentionV2Pv.U8TileScales, Int8AttentionV2Pv.F16 })
        {
            byte[]? first = null;
            foreach (Int8V2Case c in new[] { hm, tm, fused, hm })
            {
                (AttentionCall call, Tensor o) = GpuCall(c, DType.F16);
                using Tensor __ = o;
                byte[] bytes = cuda.RunInt8AttentionV2ForTest(call, pv, 128).Output;
                if (first is null) first = bytes;
                else Assert.Equal(first, bytes);
            }
        }
    }

    /// <summary>The planner's choice with the knob on: S.17 and M.4 for F16 and F32, the widened grouped call through S.17, and every
    /// call v2 does not run refused before launch (bias, head dim, BF16, grouped K/V on its own route, a non-positive scale, the
    /// engage length), with the plan unchanged with the knob off.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Rows_RouteAndRefuse()
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        List<Tensor> owned = [];
        Tensor New(TensorShape shape, DType dt)
        {
            Tensor t = new(shape, dt);
            owned.Add(t);
            return t;
        }
        AttentionCall Dense(DType dt, int d, Tensor? bias = null, float scale = 0.088f)
        {
            Tensor q = New(new TensorShape(1, 2, 256, d), dt), k = New(q.Shape, dt), o = New(q.Shape, dt);
            return AttentionCall.FromScaledDotProductAttention(o, q, k, k, bias, scale, false);
        }
        void Refused(AttentionCall call, string why)
        {
            AttentionPlan plan = cuda.PlanAttention(call);
            output.WriteLine($"{why}: {plan.Describe()}");
            Assert.NotEqual("int8.v2", plan.RowId);
            Assert.Contains($"int8.v2 ({why})", plan.Row is null ? plan.Reason : plan.Declined ?? "");
        }
        try
        {
            KnobStore.Set(EngineKnobs.Int8AttnV2MinTokens, 1);
            string off = cuda.PlanAttention(Dense(DType.F16, 128)).RowId;
            Assert.NotEqual("int8.v2", off);
            KnobStore.Set(EngineKnobs.Int8AttnV2, true);
            Assert.Equal("int8.v2", cuda.PlanAttention(Dense(DType.F16, 128)).RowId);
            Assert.Equal("int8.v2", cuda.PlanAttention(Dense(DType.F32, 128)).RowId);
            Tensor tq = New(new TensorShape(256, 256), DType.F16), to = New(tq.Shape, DType.F16), gk = New(new TensorShape(256, 128), DType.F16);
            Assert.Equal("int8.v2", cuda.PlanAttention(AttentionCall.FromTokenMajor(to, tq, tq, tq, null, 2, 2, 128, 0.088f, false)).RowId);
            Assert.Equal("permute.widen.dense",
                cuda.PlanAttention(AttentionCall.FromTokenMajor(to, tq, gk, gk, null, 2, 1, 128, 0.088f, false)).RowId);
            Refused(Dense(DType.F16, 128, New(new TensorShape(1, 1, 1, 256), DType.F32)), "Bias");
            Refused(Dense(DType.F16, 64), "HeadDim");
            Refused(Dense(DType.BF16, 128), "QueryType");
            Refused(Dense(DType.F16, 128, scale: -0.088f), "not admitted");
            AttentionCall gqa = Dense(DType.F16, 128);
            Refused(gqa with { Key = gqa.Key with { Heads = 1 }, Value = gqa.Value with { Heads = 1 } }, "GroupedKv");
            KnobStore.Set(EngineKnobs.Int8AttnV2MinTokens, 512);
            Refused(Dense(DType.F16, 128), "not admitted");
            KnobStore.Clear(EngineKnobs.Int8AttnV2);
            Assert.Equal(off, cuda.PlanAttention(Dense(DType.F16, 128)).RowId);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Int8AttnV2);
            KnobStore.Clear(EngineKnobs.Int8AttnV2MinTokens);
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    /// <summary>A step graph holding the call, replayed twice, gives the eager call's bits; the workspace sized by the eager call is not
    /// regrown under capture.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void GraphCapture_ReplaysTwice_BitwiseEqualToEager()
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        Int8V2Spec spec = new() { Id = "capture", Heads = 2, KvHeads = 2, Sq = 520, Skv = 520, HeadDim = 128, DType = DType.F32 };
        using Int8V2Case c = Int8V2Case.Create(spec);
        TensorShape shape = new(1, 2, 520, 128);
        using Tensor qh = Host(shape, c.Q), kh = Host(shape, c.K), vh = Host(shape, c.V);
        using Tensor q = new(shape, DType.F32), k = new(shape, DType.F32), v = new(shape, DType.F32);
        cuda.Scale(q, qh, 1f);
        cuda.Scale(k, kh, 1f);
        cuda.Scale(v, vh, 1f);
        KnobStore.Set(EngineKnobs.Int8AttnV2, true);
        KnobStore.Set(EngineKnobs.Int8AttnV2MinTokens, 1);
        try
        {
            using Tensor eager = new(shape, DType.F32), graphed = new(shape, DType.F32);
            float[] expected = new float[shape.ElementCount], first = new float[expected.Length], second = new float[expected.Length];
            cuda.ScaledDotProductAttention(eager, q, k, v, null, 0.088f);
            cuda.ReadResidentInto(eager, expected);
            long bytes = cuda.Int8V2WorkspaceBytes, runs = cuda.Int8AttentionV2ExecutionCount;
            cuda.StepGraphBegin();
            cuda.ScaledDotProductAttention(graphed, q, k, v, null, 0.088f);
            cuda.StepGraphEndAndLaunch();
            cuda.ReadResidentInto(graphed, first);
            cuda.StepGraphLaunch();
            cuda.ReadResidentInto(graphed, second);
            Assert.Equal(runs + 1, cuda.Int8AttentionV2ExecutionCount);
            Assert.Equal(bytes, cuda.Int8V2WorkspaceBytes);
            Assert.Equal(expected, first);
            Assert.Equal(expected, second);
            cuda.StepGraphReset();
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Int8AttnV2);
            KnobStore.Clear(EngineKnobs.Int8AttnV2MinTokens);
        }
    }

    /// <summary>The question an attention layer asks before keeping a call token-major for M.4: yes eagerly, no inside a capture the
    /// workspace does not fit (M.4 declines there, and the call goes to Sage instead), yes once an eager call has grown it.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TakesTokenMajor_IsNoInsideACaptureTheWorkspaceDoesNotFit()
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        const int Heads = 2, Tokens = 600;
        TensorShape shape = new(1, Heads, Tokens, 128);
        using Tensor q = new(shape, DType.F16), o = new(shape, DType.F16), small = new(new TensorShape(4), DType.F32);
        cuda.Scale(q, q, 1f);
        KnobStore.Set(EngineKnobs.Int8AttnV2, true);
        KnobStore.Set(EngineKnobs.Int8AttnV2MinTokens, 512);
        try
        {
            Assert.True(cuda.Int8AttentionV2TakesTokenMajor(Heads, Tokens, Tokens, 128, DType.F16));
            Assert.False(cuda.Int8AttentionV2TakesTokenMajor(Heads, Tokens, Tokens, 64, DType.F16));
            Assert.False(cuda.Int8AttentionV2TakesTokenMajor(Heads, 500, Tokens, 128, DType.F16));
            Assert.False(cuda.Int8AttentionV2TakesTokenMajor(Heads, Tokens, Tokens, 128, DType.BF16));
            bool Captured()
            {
                cuda.StepGraphBegin();
                cuda.Scale(small, small, 1f);
                bool takes = cuda.Int8AttentionV2TakesTokenMajor(Heads, Tokens, Tokens, 128, DType.F16);
                cuda.StepGraphEndAndLaunch();
                cuda.StepGraphReset();
                return takes;
            }
            Assert.Equal(0, cuda.Int8V2WorkspaceBytes);
            Assert.False(Captured());
            cuda.ScaledDotProductAttention(o, q, q, q, null, 0.088f);
            Assert.Equal(1, cuda.Int8AttentionV2ExecutionCount);
            Assert.True(Captured());
            KnobStore.Set(EngineKnobs.Int8AttnV2, false);
            Assert.False(cuda.Int8AttentionV2TakesTokenMajor(Heads, Tokens, Tokens, 128, DType.F16));
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Int8AttnV2);
            KnobStore.Clear(EngineKnobs.Int8AttnV2MinTokens);
        }
    }

    /// <summary>A capture whose workspace has not grown declines S.17, and the call runs on the next row; the grouped token-major
    /// row's yield then lands on the dense rows, where S.1 follows S.17.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void DeclinedUnderCapture_RunsTheNextRow()
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        TensorShape shape = new(1, 2, 600, 128);
        using Tensor q = new(shape, DType.F16), o = new(shape, DType.F16);
        cuda.Scale(q, q, 1f);
        KnobStore.Set(EngineKnobs.Int8AttnV2, true);
        KnobStore.Set(EngineKnobs.Int8AttnV2MinTokens, 512);
        try
        {
            cuda.StepGraphBegin();
            cuda.ScaledDotProductAttention(o, q, q, q, null, 0.088f);
            string? route = CudaBackend.LastAttentionRoute;
            cuda.StepGraphEndAndLaunch();
            cuda.StepGraphReset();
            Assert.Equal(0, cuda.Int8AttentionV2ExecutionCount);
            Assert.NotNull(route);
            Assert.NotEqual("S.17", route);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Int8AttnV2);
            KnobStore.Clear(EngineKnobs.Int8AttnV2MinTokens);
        }
    }

    /// <summary>Device time of the three launches at the CK parity shapes, for the record: stats, quant, attention.</summary>
    [Trait("Category", "GpuBenchmark")]
    [Theory]
    [InlineData(4096, "seq", 64)]
    [InlineData(8192, "seq", 64)]
    [InlineData(8192, "tile", 64)]
    [InlineData(8192, "seq", 128)]
    [InlineData(12288, "tile", 64)]
    public void StageTimes_AtCkParity(int tokens, string pvName, int blockRows)
    {
        if (!Open(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        TensorShape shape = new(1, 24, tokens, 128);
        using Tensor q = new(shape, DType.F16), k = new(shape, DType.F16), v = new(shape, DType.F16), o = new(shape, DType.F16);
        Random rng = new(5);
        foreach (Tensor t in new[] { q, k, v })
        {
            Span<Half> span = t.AsSpan<Half>();
            for (int i = 0; i < span.Length; i++) span[i] = (Half)(rng.NextDouble() * 2 - 1);
        }
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 0.088f, false);
        float[] ms = cuda.Int8V2StageTimesForTest(call, Pv(pvName), blockRows, 10);
        double flops = 4.0 * 24 * tokens * (double)tokens * 128;
        output.WriteLine($"{tokens} {pvName} BR{blockRows}: stats {ms[0]:F3} ms, quant {ms[1]:F3} ms, attention {ms[2]:F3} ms "
            + $"({flops / (ms[2] * 1e-3) / 1e12:F1} TOPS kernel, {flops / (ms.Sum() * 1e-3) / 1e12:F1} TOPS total)");
    }

    private static Tensor Host(TensorShape shape, float[] values)
    {
        Tensor t = new(shape, DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
