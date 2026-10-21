namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>Double-precision attention on a sample of (batch, head, query row) triples, from the exact input values a
/// kernel saw. Independent of every backend kernel, so it can judge all of them, including TF32 and F16 paths.</summary>
public sealed class AttentionOracle
{
    private readonly AttentionBenchShape _s;
    private readonly float[] _q, _k, _v;
    private readonly float[]? _mask;
    private readonly float _scale;

    /// <summary>Host inputs are canonical head-major: Q <c>[B,Hq,Sq,D]</c>, K/V <c>[B,Hkv,Skv,D]</c>; the mask is
    /// <c>[1,Hq,Sq,Skv]</c> (bias-head) or <c>[1,1,1,Skv]</c> (bias-key).</summary>
    public AttentionOracle(AttentionBenchShape shape, float[] q, float[] k, float[] v, float[]? mask, float scale)
    {
        _s = shape; _q = q; _k = k; _v = v; _mask = mask; _scale = scale;
    }

    /// <summary>The sampled triples: up to three heads, up to <paramref name="rows"/> query rows spread over the sequence
    /// including the first and last, first and last batch.</summary>
    public IReadOnlyList<(int B, int H, int Row)> Sample(int rows)
    {
        int[] heads = new[] { 0, _s.Hq / 2, _s.Hq - 1 }.Distinct().ToArray();
        int[] batches = new[] { 0, _s.B - 1 }.Distinct().ToArray();
        int n = Math.Min(rows, _s.Sq);
        int[] qRows = Enumerable.Range(0, n).Select(i => n == 1 ? 0 : (int)((long)i * (_s.Sq - 1) / (n - 1))).Distinct().ToArray();
        List<(int, int, int)> list = [];
        foreach (int b in batches) foreach (int h in heads) foreach (int r in qRows) list.Add((b, h, r));
        return list;
    }

    /// <summary>Reference output row <c>[D]</c> for one triple.</summary>
    public double[] Row(int b, int h, int row)
    {
        int d = _s.D, skv = _s.Skv, kvh = h / (_s.Hq / _s.Hkv);
        long qBase = (((long)b * _s.Hq + h) * _s.Sq + row) * d;
        long kvBase = ((long)b * _s.Hkv + kvh) * skv * d;
        int visible = _s.Causal ? Math.Min(skv, _s.QOffset + row + 1) : skv;
        double[] scores = new double[visible];
        double max = double.NegativeInfinity;
        for (int j = 0; j < visible; j++)
        {
            double dot = 0;
            long kOff = kvBase + (long)j * d;
            for (int c = 0; c < d; c++) dot += (double)_q[qBase + c] * _k[kOff + c];
            double sc = dot * _scale + Bias(h, row, j);
            scores[j] = sc;
            if (sc > max) max = sc;
        }
        double[] o = new double[d];
        if (double.IsNegativeInfinity(max)) return o;
        double sum = 0;
        for (int j = 0; j < visible; j++)
        {
            double p = Math.Exp(scores[j] - max);
            sum += p;
            long vOff = kvBase + (long)j * d;
            for (int c = 0; c < d; c++) o[c] += p * _v[vOff + c];
        }
        for (int c = 0; c < d; c++) o[c] /= sum;
        return o;
    }

    private double Bias(int h, int row, int j) => _s.Mask switch
    {
        "bias-head" => _mask![((long)h * _s.Sq + row) * _s.Skv + j],
        "bias-key" => _mask![j],
        _ => 0.0,
    };

    /// <summary>Relative L2 error, cosine similarity and max absolute error of <paramref name="actual"/> over the sample,
    /// with relative error floored by <c>1e-6·max|reference|</c> so near-zero rows cannot dominate.</summary>
    public (double RelL2, double Cos, double MaxAbs, int Rows) Compare(Func<int, int, int, int, double> actual, int rows)
    {
        double num = 0, den = 0, dot = 0, na = 0, maxAbs = 0, refMax = 0;
        IReadOnlyList<(int B, int H, int Row)> sample = Sample(rows);
        List<(double[] Ref, (int B, int H, int Row) T)> refs = new(sample.Count);
        foreach ((int B, int H, int Row) t in sample)
        {
            double[] r = Row(t.B, t.H, t.Row);
            foreach (double x in r) refMax = Math.Max(refMax, Math.Abs(x));
            refs.Add((r, t));
        }
        double floor = 1e-6 * refMax;
        foreach ((double[] r, (int B, int H, int Row) t) in refs)
        {
            for (int c = 0; c < r.Length; c++)
            {
                double a = actual(t.B, t.H, t.Row, c);
                double e = a - r[c];
                if (!double.IsFinite(a)) e = double.PositiveInfinity;
                num += e * e;
                den += Math.Max(r[c] * r[c], floor * floor);
                dot += a * r[c];
                na += a * a;
                maxAbs = Math.Max(maxAbs, Math.Abs(e));
            }
        }
        double rel = den > 0 ? Math.Sqrt(num / den) : double.NaN;
        double refNorm = Math.Sqrt(refs.Sum(x => x.Ref.Sum(v => v * v)));
        double cos = na > 0 && refNorm > 0 ? dot / (Math.Sqrt(na) * refNorm) : double.NaN;
        return (rel, cos, maxAbs, sample.Count);
    }
}
