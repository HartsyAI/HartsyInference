using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>How a case's Q, K and V are drawn before <see cref="Int8V2Spec.Mutate"/> and dtype rounding.</summary>
public enum Int8V2Fill { Uniform, Gaussian }

/// <summary>How a case's operands sit in memory.</summary>
public enum Int8V2Layout { HeadMajor, TokenMajor, FusedQkv }

/// <summary>An INT8 attention v2 case: geometry, inputs, mask and logit terms, and the query rows checked.</summary>
public sealed record Int8V2Spec
{
    /// <summary>Case id.</summary>
    public required string Id { get; init; }

    public int Batch { get; init; } = 1;

    public int Heads { get; init; } = 1;

    public int KvHeads { get; init; } = 1;

    public required int Sq { get; init; }

    public required int Skv { get; init; }

    public required int HeadDim { get; init; }

    /// <summary>Operand storage dtype; the values are rounded to it, so the oracle sees what the kernel sees.</summary>
    public DType DType { get; init; } = DType.F16;

    public Int8V2Fill Fill { get; init; } = Int8V2Fill.Gaussian;

    public Int8V2Layout Layout { get; init; } = Int8V2Layout.HeadMajor;

    /// <summary>Valid keys; 0 means all <see cref="Skv"/>.</summary>
    public int KvLen { get; init; }

    public int QueryOffset { get; init; }

    public bool Causal { get; init; }

    public int? WindowLeft { get; init; }

    public int? WindowRight { get; init; }

    public int BlockSize { get; init; }

    public float Softcap { get; init; }

    /// <summary>Per-head sink logits, or null.</summary>
    public float[]? Sinks { get; init; }

    /// <summary>Per-key additive bias broadcast over batches, heads and rows (−∞ excludes), or null.</summary>
    public float[]? KeyBias { get; init; }

    /// <summary>Score scale; 0 means 1/√HeadDim.</summary>
    public float Scale { get; init; }

    /// <summary>Query rows checked: evenly spaced, first and last included; 0 checks every row.</summary>
    public int SampledRows { get; init; } = 64;

    public int Seed { get; init; } = 1;

    /// <summary>Rewrites head-major Q <c>[B,H,Sq,D]</c>, K and V <c>[B,Hkv,Skv,D]</c> after they are drawn.</summary>
    public Action<float[], float[], float[]>? Mutate { get; init; }
}

/// <summary>One INT8 attention v2 case built from a <see cref="Int8V2Spec"/>: operand tensors, the call over them, the rows
/// checked, and the emulator and the double-precision contract oracle on those rows.</summary>
public sealed class Int8V2Case : IDisposable
{
    private readonly List<Tensor> _owned = [];

    private Int8V2Case(Int8V2Spec spec) => Spec = spec;

    public Int8V2Spec Spec { get; }

    /// <summary>Head-major values after dtype rounding: Q <c>[B,H,Sq,D]</c>, K and V <c>[B,Hkv,Skv,D]</c>.</summary>
    public float[] Q { get; private set; } = [];

    public float[] K { get; private set; } = [];

    public float[] V { get; private set; } = [];

    /// <summary>The call; its output is an F32 tensor the emulator writes.</summary>
    public AttentionCall Call { get; private set; }

    /// <summary>The query rows checked, ascending.</summary>
    public int[] Rows { get; private set; } = [];

    /// <summary>Checked output values per (batch, head, row): the row width is the head dim.</summary>
    public int RowWidth => Spec.HeadDim;

    /// <summary>Draws the spec's inputs and builds the case.</summary>
    public static Int8V2Case Create(Int8V2Spec spec)
    {
        Random rng = new(spec.Seed);
        float[] q = Draw(rng, spec.Fill, spec.Batch * spec.Heads * spec.Sq * spec.HeadDim);
        float[] k = Draw(rng, spec.Fill, spec.Batch * spec.KvHeads * spec.Skv * spec.HeadDim);
        float[] v = Draw(rng, spec.Fill, k.Length);
        spec.Mutate?.Invoke(q, k, v);
        return FromArrays(spec, q, k, v);
    }

    /// <summary>A case over the given head-major values, rounded to the spec's dtype.</summary>
    public static Int8V2Case FromArrays(Int8V2Spec spec, float[] q, float[] k, float[] v)
    {
        Int8V2Case c = new(spec);
        c.Build((float[])q.Clone(), (float[])k.Clone(), (float[])v.Clone());
        return c;
    }

    /// <summary>The emulator's output on the checked rows, <c>[B, Hq, rows, D]</c>; <paramref name="lse"/> receives
    /// <c>[B, Hq, Sq]</c> when given.</summary>
    public float[] Emulate(Int8AttentionV2Options options, Span<float> lse = default)
    {
        Int8AttentionV2Reference.Run(Call, options, Rows, lse);
        return CheckedRows(Call.Output);
    }

    /// <summary>The emulator's output on the checked rows from already quantized operands.</summary>
    public float[] Attend(Int8AttentionV2Operands operands, Span<float> lse = default)
    {
        Int8AttentionV2Reference.Attend(Call, operands, lse);
        return CheckedRows(Call.Output);
    }

    /// <summary>The contract oracle (<see cref="AttentionHostReference"/>, double precision) on the checked rows; one call per row
    /// so long sequences stay affordable.</summary>
    public float[] Oracle(Span<float> lse = default)
    {
        AttentionCall call = Call;
        using Tensor o = new(call.Output.Storage.Shape, DType.F32);
        AttentionOperand output = call.Output with { Storage = o };
        float[] rowLse = new float[call.Batch * call.Heads];
        foreach (int r in Rows)
        {
            AttentionCall one = call with
            {
                Query = call.Query with { Tokens = 1, Offset = call.Query.Offset + r * call.Query.TokenStride },
                Output = output with { Tokens = 1, Offset = output.Offset + r * output.TokenStride },
                Positions = AttentionPositions.Host(call.ValidKvLength, call.Positions.QueryOffset + r),
            };
            AttentionHostReference.Run(one, rowLse);
            if (!lse.IsEmpty)
                for (int bh = 0; bh < rowLse.Length; bh++) lse[bh * call.QueryTokens + r] = rowLse[bh];
        }
        return CheckedRows(output);
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    private void Build(float[] q, float[] k, float[] v)
    {
        Int8V2Spec s = Spec;
        Round(q, s.DType);
        Round(k, s.DType);
        Round(v, s.DType);
        (Q, K, V) = (q, k, v);
        int d = s.HeadDim;
        AttentionOperand qo, ko, vo;
        if (s.Layout == Int8V2Layout.FusedQkv)
        {
            if (s.Sq != s.Skv) throw new ArgumentException("A fused QKV case attends its own tokens: Sq must equal Skv.");
            int width = (s.Heads + 2 * s.KvHeads) * d;
            float[] rows = new float[s.Batch * s.Sq * width];
            for (int b = 0; b < s.Batch; b++)
                for (int t = 0; t < s.Sq; t++)
                {
                    int row = (b * s.Sq + t) * width;
                    for (int h = 0; h < s.Heads; h++) q.AsSpan(((b * s.Heads + h) * s.Sq + t) * d, d).CopyTo(rows.AsSpan(row + h * d));
                    for (int h = 0; h < s.KvHeads; h++)
                    {
                        k.AsSpan(((b * s.KvHeads + h) * s.Skv + t) * d, d).CopyTo(rows.AsSpan(row + (s.Heads + h) * d));
                        v.AsSpan(((b * s.KvHeads + h) * s.Skv + t) * d, d).CopyTo(rows.AsSpan(row + (s.Heads + s.KvHeads + h) * d));
                    }
                }
            Tensor fused = Own(new TensorShape(s.Batch * s.Sq, width), rows, s.DType);
            qo = AttentionOperand.Packed(fused, s.Sq, width, 0, s.Heads, d, s.Batch);
            ko = AttentionOperand.Packed(fused, s.Skv, width, s.Heads * d, s.KvHeads, d, s.Batch);
            vo = AttentionOperand.Packed(fused, s.Skv, width, (s.Heads + s.KvHeads) * d, s.KvHeads, d, s.Batch);
        }
        else if (s.Layout == Int8V2Layout.TokenMajor)
        {
            qo = TokenMajor(q, s.Heads, s.Sq);
            ko = TokenMajor(k, s.KvHeads, s.Skv);
            vo = TokenMajor(v, s.KvHeads, s.Skv);
        }
        else
        {
            qo = AttentionOperand.HeadMajor(Own(new TensorShape(s.Batch, s.Heads, s.Sq, d), q, s.DType));
            ko = AttentionOperand.HeadMajor(Own(new TensorShape(s.Batch, s.KvHeads, s.Skv, d), k, s.DType));
            vo = AttentionOperand.HeadMajor(Own(new TensorShape(s.Batch, s.KvHeads, s.Skv, d), v, s.DType));
        }
        AttentionOperand oo = AttentionOperand.HeadMajor(
            Own(new TensorShape(s.Batch, s.Heads, s.Sq, d), new float[s.Batch * s.Heads * s.Sq * d], DType.F32));
        Tensor? bias = s.KeyBias is null ? null : Own(new TensorShape(1, 1, 1, s.Skv), s.KeyBias, DType.F32);
        Tensor? sinks = s.Sinks is null ? null : Own(new TensorShape(s.Heads), s.Sinks, DType.F32);
        Call = new AttentionCall
        {
            Query = qo, Key = ko, Value = vo, Output = oo,
            Scale = s.Scale != 0f ? s.Scale : 1f / MathF.Sqrt(d),
            Mask = new AttentionMask
            {
                Causal = s.Causal, WindowLeft = s.WindowLeft, WindowRight = s.WindowRight, BlockSize = s.BlockSize, Bias = bias,
            },
            Positions = AttentionPositions.Host(s.KvLen > 0 ? s.KvLen : s.Skv, s.QueryOffset),
            Logits = new AttentionLogitTerms(s.Softcap, sinks, null),
        };
        int n = s.SampledRows <= 0 || s.SampledRows >= s.Sq ? s.Sq : s.SampledRows;
        Rows = n == s.Sq ? Enumerable.Range(0, s.Sq).ToArray()
            : Enumerable.Range(0, n).Select(i => n == 1 ? 0 : (int)((long)i * (s.Sq - 1) / (n - 1))).Distinct().ToArray();
    }

    private AttentionOperand TokenMajor(float[] headMajor, int heads, int seq)
    {
        int b = Spec.Batch, d = Spec.HeadDim;
        float[] t = new float[headMajor.Length];
        for (int bi = 0; bi < b; bi++)
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < seq; s++)
                    headMajor.AsSpan(((bi * heads + h) * seq + s) * d, d).CopyTo(t.AsSpan(((bi * seq + s) * heads + h) * d));
        return AttentionOperand.TokenMajor(Own(new TensorShape(b, seq, heads, d), t, Spec.DType), heads, d);
    }

    private float[] CheckedRows(in AttentionOperand output)
    {
        float[] all = output.Storage.AsReadOnlySpan<float>().ToArray();
        int d = Spec.HeadDim;
        float[] rows = new float[Spec.Batch * Spec.Heads * Rows.Length * d];
        int i = 0;
        for (int b = 0; b < Spec.Batch; b++)
            for (int h = 0; h < Spec.Heads; h++)
                foreach (int r in Rows)
                {
                    all.AsSpan((int)(b * output.BatchStride + h * output.HeadStride + r * output.TokenStride), d).CopyTo(rows.AsSpan(i));
                    i += d;
                }
        return rows;
    }

    private Tensor Own(TensorShape shape, float[] values, DType dtype)
    {
        Tensor f32 = new(shape, DType.F32);
        values.CopyTo(f32.AsSpan<float>());
        if (dtype == DType.F32)
        {
            _owned.Add(f32);
            return f32;
        }
        Tensor narrow = f32.CastTo(dtype);
        f32.Dispose();
        _owned.Add(narrow);
        return narrow;
    }

    private static float[] Draw(Random rng, Int8V2Fill fill, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++)
            a[i] = fill == Int8V2Fill.Uniform ? (float)(rng.NextDouble() * 2 - 1)
                : (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        return a;
    }

    private static void Round(float[] a, DType dtype)
    {
        if (dtype == DType.F16)
            for (int i = 0; i < a.Length; i++) a[i] = (float)(Half)a[i];
        else if (dtype == DType.BF16)
            for (int i = 0; i < a.Length; i++)
            {
                uint u = BitConverter.SingleToUInt32Bits(a[i]);
                a[i] = BitConverter.UInt32BitsToSingle(((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16) << 16);
            }
    }
}

/// <summary>Agreement measures over checked rows, as the Phase-0 shape table defines them.</summary>
public static class Int8V2Measure
{
    /// <summary>CK's published bar, adopted before any v2 kernel existed: relL2 and global cosine.</summary>
    public const double BarRelL2 = 6e-3, BarCos = 0.99998;

    /// <summary>Relative L2 with each element's reference magnitude floored at 1e-6·max|ref|, global cosine, max |Δ|. Any NaN
    /// makes every measure NaN.</summary>
    public static (double RelL2, double Cos, double MaxAbs) Of(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        if (actual.Length != expected.Length) throw new ArgumentException($"{actual.Length} values against {expected.Length}.");
        double refMax = 0;
        foreach (float e in expected) refMax = Math.Max(refMax, Math.Abs((double)e));
        double floor = 1e-6 * refMax, num = 0, den = 0, dot = 0, na = 0, ne = 0, max = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double a = actual[i], e = expected[i], diff = a - e;
            num += diff * diff;
            den += Math.Max(e * e, floor * floor);
            dot += a * e;
            na += a * a;
            ne += e * e;
            max = double.IsNaN(diff) ? double.NaN : Math.Max(max, Math.Abs(diff));
        }
        double rel = den > 0 ? Math.Sqrt(num / den) : num == 0 ? 0 : double.PositiveInfinity;
        double cos = na > 0 && ne > 0 ? dot / Math.Sqrt(na * ne) : na == ne ? 1 : 0;
        return (rel, cos, max);
    }
}

/// <summary>Phase-0 attention shapes (benchmarks/python-baseline/attention_shapes.json) the accuracy table samples.</summary>
public static class Int8V2Shapes
{
    /// <summary>(id, query heads, sq, skv, head dim, causal) for the shapes INT8 v2 serves.</summary>
    public static readonly (string Id, int Heads, int Sq, int Skv, int HeadDim, bool Causal)[] Phase0 =
    [
        ("krea2-1024", 48, 4108, 4108, 128, false),
        ("flux-1024", 24, 4608, 4608, 128, false),
        ("wan14b-480p", 40, 32760, 32760, 128, false),
        ("wan14b-xattn", 40, 32760, 512, 128, false),
        ("ltx25-video", 32, 17480, 17480, 128, false),
        ("ltx25-a2v", 32, 151, 17480, 64, false),
        ("d40-sd15", 8, 4096, 4096, 40, false),
        ("d64-sd35m", 24, 4429, 4429, 64, false),
        ("d96-lumina2", 24, 4134, 4134, 96, false),
        ("d256-ideogram4", 18, 4335, 4335, 256, false),
        ("ck-parity-4k", 24, 4096, 4096, 128, false),
        ("llm-prefill-4k", 32, 4096, 4096, 128, true),
    ];

    /// <summary>Single-head cases for query heads 0, H/2 and H−1 of a shape (independent draws per head, as the GPU table samples
    /// heads), each checking up to 64 rows.</summary>
    public static IEnumerable<Int8V2Spec> HeadsOf((string Id, int Heads, int Sq, int Skv, int HeadDim, bool Causal) shape, Int8V2Fill fill)
    {
        foreach (int h in new SortedSet<int> { 0, shape.Heads / 2, shape.Heads - 1 })
            yield return new Int8V2Spec
            {
                Id = $"{shape.Id}-h{h}-{fill.ToString().ToLowerInvariant()}", Sq = shape.Sq, Skv = shape.Skv, HeadDim = shape.HeadDim,
                Causal = shape.Causal, QueryOffset = shape.Causal ? shape.Skv - shape.Sq : 0, Fill = fill,
                Seed = AttnCase.StableHash(shape.Id) + h,
            };
    }
}

/// <summary>The adversarial INT8 attention v2 cases (Phase 6 item 3).</summary>
public static class Int8V2Adversarial
{
    /// <summary>The score gap below a dominant key that the sub-threshold cases put their tail at: −ln 600.</summary>
    public static readonly float TailGap = MathF.Log(600f);

    /// <summary>Every adversarial case at its table size.</summary>
    public static IEnumerable<Int8V2Spec> All()
    {
        yield return Outliers("k-channel-outliers", (_, k, _, s) => AddToChannels(k, s.KvHeads * s.Skv, s.HeadDim, [5, 37, 90], 20f));
        yield return Outliers("k-token-outliers", (_, k, _, s) => ScaleRows(k, s.HeadDim, [100, 777, 1500, 2001], 20f));
        yield return Outliers("v-channel-outliers", (_, _, v, s) => ScaleChannels(v, s.KvHeads * s.Skv, s.HeadDim, [3, 64], 50f));
        yield return Outliers("v-token-outliers", (_, _, v, s) => ScaleRows(v, s.HeadDim, [50, 900], 50f));
        yield return new Int8V2Spec
        {
            Id = "v-above-f16-max", Sq = 256, Skv = 1024, HeadDim = 64, DType = DType.F32, Seed = 11,
            Mutate = (_, _, v) => { for (int i = 0; i < v.Length; i++) v[i] *= 1e5f; },
        };
        yield return new Int8V2Spec
        {
            Id = "uniform-attention-16k", Sq = 64, Skv = 16384, HeadDim = 64, Seed = 12, Mutate = (q, _, _) => Array.Clear(q),
        };
        yield return SingleDominantKey();
        yield return SubThresholdSpread(dominantAt: 0);
        yield return SubThresholdSpread(dominantAt: 8192);
        yield return SubThresholdOneTile(keys: 128);
        yield return FiveThousandNear1e4();
    }

    /// <summary>Gaussian 1×1×256×2048×128 with <paramref name="mutate"/> applied.</summary>
    public static Int8V2Spec Outliers(string id, Action<float[], float[], float[], Int8V2Spec> mutate)
    {
        Int8V2Spec spec = new() { Id = id, Sq = 256, Skv = 2048, HeadDim = 128, Seed = AttnCase.StableHash(id) };
        return spec with { Mutate = (q, k, v) => mutate(q, k, v, spec) };
    }

    /// <summary>Every query row is one vector u of norm √D and key 1234 is 12·u/√D, a score of 12 against N(0,1) for the rest:
    /// that key holds 98% of every row.</summary>
    public static Int8V2Spec SingleDominantKey() => new()
    {
        Id = "single-dominant-key", Sq = 128, Skv = 2048, HeadDim = 128, Seed = 13,
        Mutate = (q, k, _) =>
        {
            const int d = 128;
            float[] u = q.AsSpan(0, d).ToArray();
            float norm = MathF.Sqrt(u.Sum(x => x * x));
            for (int c = 0; c < d; c++) u[c] *= MathF.Sqrt(d) / norm;
            for (int r = 0; r < q.Length / d; r++) u.CopyTo(q.AsSpan(r * d));
            for (int c = 0; c < d; c++) k[1234 * d + c] = 12f * u[c] / MathF.Sqrt(d);
        },
    };

    /// <summary>One dominant key at <paramref name="dominantAt"/> and 8,192 tail keys −ln 600 below it: Q is one-hot, the
    /// dominant key's V is 0 and every tail key's V is 1, so the exact output is the tail's share of the mass, 0.932.</summary>
    public static Int8V2Spec SubThresholdSpread(int dominantAt, int rows = 1) => OneHot($"sub-threshold-8192@{dominantAt}", 8193,
        dominantAt, rows);

    /// <summary>The dominant key and its <paramref name="keys"/> − 1 tail keys in one key tile (Skv ≤ BC): every tail key sits
    /// below 1/510 of the tile's maximum, so u8 P rounds the whole tail to zero.</summary>
    public static Int8V2Spec SubThresholdOneTile(int keys, int rows = 256) => OneHot($"sub-threshold-one-tile-{keys}", keys, 0, rows);

    /// <summary>SageAttention2's argument for a floating P: 5,040 keys near probability 1e-4 hold half the mass, and every key
    /// tile also holds one head key carrying ~120× more (scores set by a key bias over Q = K = 0).</summary>
    public static Int8V2Spec FiveThousandNear1e4()
    {
        const int skv = 5120;
        float[] bias = new float[skv];
        Random rng = new(14);
        for (int j = 0; j < skv; j++) bias[j] = j % 64 == 0 ? 0f : MathF.Log(0.00813f * (0.9f + 0.2f * (float)rng.NextDouble()));
        return new Int8V2Spec
        {
            Id = "five-thousand-near-1e-4", Sq = 16, Skv = skv, HeadDim = 64, Seed = 14, KeyBias = bias,
            Mutate = (q, k, _) => { Array.Clear(q); Array.Clear(k); },
        };
    }

    private static Int8V2Spec OneHot(string id, int skv, int dominantAt, int rows) => new()
    {
        Id = id, Sq = rows, Skv = skv, HeadDim = 64, DType = DType.F32, Scale = TailGap, SampledRows = 0, Seed = 15,
        Mutate = (q, k, v) =>
        {
            const int d = 64;
            Array.Clear(q);
            Array.Clear(k);
            for (int r = 0; r < rows; r++) q[r * d] = 1f;
            k[dominantAt * d] = 1f;
            Array.Fill(v, 1f);
            v.AsSpan(dominantAt * d, d).Clear();
        },
    };

    private static void AddToChannels(float[] x, int rows, int d, int[] channels, float offset)
    {
        for (int r = 0; r < rows; r++)
            foreach (int c in channels) x[r * d + c] += offset;
    }

    private static void ScaleChannels(float[] x, int rows, int d, int[] channels, float factor)
    {
        for (int r = 0; r < rows; r++)
            foreach (int c in channels) x[r * d + c] *= factor;
    }

    private static void ScaleRows(float[] x, int d, int[] rows, float factor)
    {
        foreach (int r in rows)
            for (int c = 0; c < d; c++) x[r * d + c] *= factor;
    }
}
