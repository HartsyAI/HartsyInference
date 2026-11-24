namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>A float64 transcription of the SenseNova-Loop decoder, written from <c>modeling_sensenova_loop.py</c> (checkpoint sha 3cc220e) and
/// transformers 5.18's <c>modeling_qwen3_5.py</c> and <c>_compute_yarn_parameters</c>, independent of the engine's ops: plain loops over
/// <c>double</c> arrays, every execution slot recomputed from the whole sequence (so its state is its own by construction).
/// It stands in for the torch module until a torch run is available; <c>tests/python-reference/sensenova_loop</c> holds the dumper that
/// checks this transcription against the real module.</summary>
internal sealed class SenseNovaLoopReference
{
    public const int Hidden = 32, Vocab = 40, Ffn = 24;
    public const int Heads = 4, KvHeads = 2, HeadDim = 8, RotaryDim = 4;
    public const int KeyHeads = 2, ValueHeads = 4, KeyDim = 8, ValueDim = 8, ConvKernel = 4;
    public const int SinkhornIterations = 20;
    public const double Eps = 1e-6, RopeTheta = 10000.0, YarnFactor = 2.0, YarnOriginal = 16;

    /// <summary>Per physical layer: true = linear attention.</summary>
    public static readonly bool[] Linear = [true, true, false, false];

    public int[] Plan { get; init; } = [0, 1, 2, 1, 2, 3];
    public int MhcStart { get; init; } = 1;
    public int MhcEnd { get; init; } = 3;
    public double PreScale { get; init; } = 1.0;
    public int Streams { get; init; } = 2;

    private readonly Dictionary<string, float[]> _w;
    private readonly double[] _invFreq;
    private readonly double _attentionScaling;

    public SenseNovaLoopReference(Dictionary<string, float[]> weights)
    {
        _w = weights;
        (_invFreq, _attentionScaling) = Yarn();
    }

    /// <summary>transformers 5.18 <c>_compute_yarn_parameters</c> for dim <see cref="RotaryDim"/>, truncate on.</summary>
    private static (double[], double) Yarn()
    {
        int dim = RotaryDim;
        double[] plain = new double[dim / 2];
        for (int k = 0; k < plain.Length; k++) plain[k] = 1.0 / Math.Pow(RopeTheta, (double)(2 * k) / dim);
        double CorrectionDim(double rotations) => dim * Math.Log(YarnOriginal / (rotations * 2 * Math.PI)) / (2 * Math.Log(RopeTheta));
        double low = Math.Max(Math.Floor(CorrectionDim(32)), 0), high = Math.Min(Math.Ceiling(CorrectionDim(1)), dim - 1);
        if (low == high) high += 0.001;
        double[] inv = new double[plain.Length];
        for (int k = 0; k < inv.Length; k++)
        {
            double ramp = Math.Clamp((k - low) / (high - low), 0, 1);
            double extrapolationFactor = 1 - ramp;
            inv[k] = plain[k] / YarnFactor * (1 - extrapolationFactor) + plain[k] * extrapolationFactor;
        }
        return (inv, 0.1 * Math.Log(YarnFactor) + 1.0);
    }

    public double[][] Forward(int[] ids)
    {
        int seq = ids.Length;
        double[][] h = new double[seq][];
        for (int t = 0; t < seq; t++) h[t] = Row("model.language_model.embed_tokens.weight", ids[t], Hidden);
        double[][][]? streams = null;
        foreach (int physical in Plan)
        {
            bool mhc = physical >= MhcStart && physical < MhcEnd;
            if (!mhc)
            {
                double[][] attn = Mixer(physical, h);
                double[][] afterAttn = Add(h, attn);
                h = Add(afterAttn, Mlp(physical, afterAttn));
                continue;
            }
            if (physical == MhcStart)
            {
                streams = new double[seq][][];
                for (int t = 0; t < seq; t++)
                {
                    streams[t] = new double[Streams][];
                    for (int s = 0; s < Streams; s++) streams[t][s] = (double[])h[t].Clone();
                }
            }
            string layer = $"model.language_model.layers.{physical}";
            streams = MhcSublayer(streams!, $"{layer}.mhc_attn", branchIn => Mixer(physical, branchIn));
            streams = MhcSublayer(streams, $"{layer}.mhc_ffn", branchIn => Mlp(physical, branchIn));
            if (physical == MhcEnd - 1)
            {
                for (int t = 0; t < seq; t++)
                    for (int c = 0; c < Hidden; c++)
                    {
                        double sum = 0;
                        for (int s = 0; s < Streams; s++) sum += streams[t][s][c];
                        h[t][c] = sum / Streams;
                    }
            }
        }
        double[][] normed = new double[seq][];
        double[] w = Vec("model.language_model.norm.weight");
        for (int t = 0; t < seq; t++) normed[t] = RmsNormOnePlus(h[t], w);
        return normed;
    }

    // ---- hyper-connection --------------------------------------------------------------------------------------------------

    private double[][][] MhcSublayer(double[][][] r, string router, Func<double[][], double[][]> sublayer)
    {
        int seq = r.Length;
        double[] weight = Vec($"{router}.weight"), bias = Vec($"{router}.bias");
        int mix = 2 * Streams + Streams * Streams;
        double[][] branchIn = new double[seq][];
        double[][] post = new double[seq][];
        double[][][,] mixes = new double[seq][][,];
        for (int t = 0; t < seq; t++)
        {
            double[] flat = r[t].SelectMany(x => x).ToArray();
            double ms = flat.Sum(x => x * x) / flat.Length;
            double scale = 1.0 / Math.Sqrt(ms + 1e-6);
            double[] logits = new double[mix];
            for (int o = 0; o < mix; o++)
            {
                double acc = bias[o];
                for (int i = 0; i < flat.Length; i++) acc += weight[o * flat.Length + i] * flat[i] * scale;
                logits[o] = acc;
            }
            double[] pre = new double[Streams];
            post[t] = new double[Streams];
            for (int s = 0; s < Streams; s++)
            {
                pre[s] = Sigmoid(logits[s]);
                post[t][s] = 2 * Sigmoid(logits[Streams + s]);
            }
            double[,] m = new double[Streams, Streams];
            for (int i = 0; i < Streams; i++)
                for (int j = 0; j < Streams; j++) m[i, j] = logits[2 * Streams + i * Streams + j];
            mixes[t] = [Sinkhorn(m)];
            branchIn[t] = new double[Hidden];
            for (int c = 0; c < Hidden; c++)
            {
                double sum = 0;
                for (int s = 0; s < Streams; s++) sum += pre[s] * r[t][s][c];
                branchIn[t][c] = sum * PreScale;
            }
        }
        double[][] branch = sublayer(branchIn);
        double[][][] next = new double[seq][][];
        for (int t = 0; t < seq; t++)
        {
            double[,] m = mixes[t][0];
            next[t] = new double[Streams][];
            for (int i = 0; i < Streams; i++)
            {
                next[t][i] = new double[Hidden];
                for (int c = 0; c < Hidden; c++)
                {
                    double sum = 0;
                    for (int j = 0; j < Streams; j++) sum += m[i, j] * r[t][j][c];   // matmul(mix, residual)
                    next[t][i][c] = sum + post[t][i] * branch[t][c];
                }
            }
        }
        return next;
    }

    private static double[,] Sinkhorn(double[,] logits)
    {
        int n = logits.GetLength(0);
        double[,] x = (double[,])logits.Clone();
        for (int it = 0; it < SinkhornIterations; it++)
        {
            for (int i = 0; i < n; i++)   // log-softmax over the last dim (rows)
            {
                double lse = LogSumExp(Enumerable.Range(0, n).Select(j => x[i, j]));
                for (int j = 0; j < n; j++) x[i, j] -= lse;
            }
            for (int j = 0; j < n; j++)   // over dim -2 (columns)
            {
                double lse = LogSumExp(Enumerable.Range(0, n).Select(i => x[i, j]));
                for (int i = 0; i < n; i++) x[i, j] -= lse;
            }
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) x[i, j] = Math.Exp(x[i, j]);
        return x;
    }

    private static double LogSumExp(IEnumerable<double> values)
    {
        double[] v = values.ToArray();
        double max = v.Max();
        return max + Math.Log(v.Sum(x => Math.Exp(x - max)));
    }

    // ---- sublayers ---------------------------------------------------------------------------------------------------------

    private double[][] Mixer(int physical, double[][] x)
    {
        string layer = $"model.language_model.layers.{physical}";
        double[] norm = Vec($"{layer}.input_layernorm.weight");
        double[][] n = x.Select(r => RmsNormOnePlus(r, norm)).ToArray();
        return Linear[physical] ? LinearAttention(layer, n) : FullAttention(layer, n);
    }

    private double[][] Mlp(int physical, double[][] x)
    {
        string layer = $"model.language_model.layers.{physical}";
        double[] norm = Vec($"{layer}.post_attention_layernorm.weight");
        return x.Select(r =>
        {
            double[] n = RmsNormOnePlus(r, norm);
            double[] gate = MatVec($"{layer}.mlp.gate_proj.weight", n, Ffn), up = MatVec($"{layer}.mlp.up_proj.weight", n, Ffn);
            double[] act = new double[Ffn];
            for (int i = 0; i < Ffn; i++) act[i] = Silu(gate[i]) * up[i];
            return MatVec($"{layer}.mlp.down_proj.weight", act, Hidden);
        }).ToArray();
    }

    private double[][] FullAttention(string layer, double[][] x)
    {
        int seq = x.Length;
        double[] qNorm = Vec($"{layer}.self_attn.q_norm.weight"), kNorm = Vec($"{layer}.self_attn.k_norm.weight");
        double[][,] q = new double[seq][,], k = new double[seq][,], v = new double[seq][,], gate = new double[seq][,];
        for (int t = 0; t < seq; t++)
        {
            double[] qg = MatVec($"{layer}.self_attn.q_proj.weight", x[t], Heads * HeadDim * 2);
            double[] kp = MatVec($"{layer}.self_attn.k_proj.weight", x[t], KvHeads * HeadDim);
            double[] vp = MatVec($"{layer}.self_attn.v_proj.weight", x[t], KvHeads * HeadDim);
            q[t] = new double[Heads, HeadDim];
            gate[t] = new double[Heads, HeadDim];
            k[t] = new double[KvHeads, HeadDim];
            v[t] = new double[KvHeads, HeadDim];
            for (int hh = 0; hh < Heads; hh++)
            {
                double[] qh = RmsNormOnePlus(Enumerable.Range(0, HeadDim).Select(d => qg[hh * HeadDim * 2 + d]).ToArray(), qNorm);
                for (int d = 0; d < HeadDim; d++)
                {
                    q[t][hh, d] = qh[d];
                    gate[t][hh, d] = qg[hh * HeadDim * 2 + HeadDim + d];   // the gate is the second half of each head's row
                }
                Rotate(q[t], hh, t);
            }
            for (int hh = 0; hh < KvHeads; hh++)
            {
                double[] kh = RmsNormOnePlus(Enumerable.Range(0, HeadDim).Select(d => kp[hh * HeadDim + d]).ToArray(), kNorm);
                for (int d = 0; d < HeadDim; d++)
                {
                    k[t][hh, d] = kh[d];
                    v[t][hh, d] = vp[hh * HeadDim + d];
                }
                Rotate(k[t], hh, t);
            }
        }
        double[][] outp = new double[seq][];
        for (int t = 0; t < seq; t++)
        {
            double[] attnFlat = new double[Heads * HeadDim];
            for (int hh = 0; hh < Heads; hh++)
            {
                int kv = hh / (Heads / KvHeads);
                double[] score = new double[t + 1];
                for (int u = 0; u <= t; u++)
                {
                    double dot = 0;
                    for (int d = 0; d < HeadDim; d++) dot += q[t][hh, d] * k[u][kv, d];
                    score[u] = dot / Math.Sqrt(HeadDim);
                }
                double max = score.Max(), sum = score.Sum(s => Math.Exp(s - max));
                for (int d = 0; d < HeadDim; d++)
                {
                    double acc = 0;
                    for (int u = 0; u <= t; u++) acc += Math.Exp(score[u] - max) / sum * v[u][kv, d];
                    attnFlat[hh * HeadDim + d] = acc * Sigmoid(gate[t][hh, d]);
                }
            }
            outp[t] = MatVec($"{layer}.self_attn.o_proj.weight", attnFlat, Hidden);
        }
        return outp;
    }

    /// <summary>rotate_half over the first <see cref="RotaryDim"/> channels with YaRN frequencies; cos and sin carry the attention scaling.</summary>
    private void Rotate(double[,] heads, int head, int position)
    {
        int half = RotaryDim / 2;
        double[] rotated = new double[RotaryDim];
        for (int d = 0; d < RotaryDim; d++)
        {
            double angle = position * _invFreq[d % half];
            double cos = Math.Cos(angle) * _attentionScaling, sin = Math.Sin(angle) * _attentionScaling;
            double partner = d < half ? -heads[head, d + half] : heads[head, d - half];
            rotated[d] = heads[head, d] * cos + partner * sin;
        }
        for (int d = 0; d < RotaryDim; d++) heads[head, d] = rotated[d];
    }

    private double[][] LinearAttention(string layer, double[][] x)
    {
        int seq = x.Length, convDim = 2 * KeyHeads * KeyDim + ValueHeads * ValueDim, keyWidth = KeyHeads * KeyDim, valueWidth = ValueHeads * ValueDim;
        double[][] mixed = new double[seq][], z = new double[seq][], b = new double[seq][], a = new double[seq][];
        for (int t = 0; t < seq; t++)
        {
            mixed[t] = MatVec($"{layer}.linear_attn.in_proj_qkv.weight", x[t], convDim);
            z[t] = MatVec($"{layer}.linear_attn.in_proj_z.weight", x[t], valueWidth);
            b[t] = MatVec($"{layer}.linear_attn.in_proj_b.weight", x[t], ValueHeads);
            a[t] = MatVec($"{layer}.linear_attn.in_proj_a.weight", x[t], ValueHeads);
        }
        float[] conv = _w[$"{layer}.linear_attn.conv1d.weight"];   // [convDim, 1, kernel]
        double[][] convolved = new double[seq][];
        for (int t = 0; t < seq; t++)
        {
            convolved[t] = new double[convDim];
            for (int c = 0; c < convDim; c++)
            {
                double acc = 0;
                for (int j = 0; j < ConvKernel; j++)
                {
                    int src = t - (ConvKernel - 1) + j;
                    if (src >= 0) acc += conv[c * ConvKernel + j] * mixed[src][c];
                }
                convolved[t][c] = Silu(acc);
            }
        }
        double[] aLog = Vec($"{layer}.linear_attn.A_log"), dtBias = Vec($"{layer}.linear_attn.dt_bias"), normW = Vec($"{layer}.linear_attn.norm.weight");
        int ratio = ValueHeads / KeyHeads;
        double[][,] state = new double[ValueHeads][,];
        for (int hh = 0; hh < ValueHeads; hh++) state[hh] = new double[KeyDim, ValueDim];
        double[][] outp = new double[seq][];
        for (int t = 0; t < seq; t++)
        {
            double[] o = new double[valueWidth];
            for (int hh = 0; hh < ValueHeads; hh++)
            {
                int kh = hh / ratio;   // repeat_interleave: value head h reads key head h / ratio
                double[] q = L2Norm(Slice(convolved[t], kh * KeyDim, KeyDim)), k = L2Norm(Slice(convolved[t], keyWidth + kh * KeyDim, KeyDim));
                double[] v = Slice(convolved[t], 2 * keyWidth + hh * ValueDim, ValueDim);
                for (int d = 0; d < KeyDim; d++) q[d] *= 1.0 / Math.Sqrt(KeyDim);
                double beta = Sigmoid(b[t][hh]);
                double g = -Math.Exp(aLog[hh]) * SoftPlus(a[t][hh] + dtBias[hh]);
                double decay = Math.Exp(g);
                double[] mem = new double[ValueDim];
                for (int i = 0; i < KeyDim; i++)
                    for (int j = 0; j < ValueDim; j++)
                    {
                        state[hh][i, j] *= decay;
                        mem[j] += state[hh][i, j] * k[i];
                    }
                for (int i = 0; i < KeyDim; i++)
                    for (int j = 0; j < ValueDim; j++) state[hh][i, j] += k[i] * (v[j] - mem[j]) * beta;
                double[] head = new double[ValueDim];
                for (int i = 0; i < KeyDim; i++)
                    for (int j = 0; j < ValueDim; j++) head[j] += state[hh][i, j] * q[i];
                double ms = head.Sum(e => e * e) / ValueDim, scale = 1.0 / Math.Sqrt(ms + Eps);
                for (int j = 0; j < ValueDim; j++)
                    o[hh * ValueDim + j] = normW[j] * head[j] * scale * Silu(z[t][hh * ValueDim + j]);
            }
            outp[t] = MatVec($"{layer}.linear_attn.out_proj.weight", o, Hidden);
        }
        return outp;
    }

    // ---- helpers -----------------------------------------------------------------------------------------------------------

    private double[] Vec(string name) => _w[name].Select(f => (double)f).ToArray();

    private double[] Row(string name, int row, int width) => Enumerable.Range(0, width).Select(c => (double)_w[name][row * width + c]).ToArray();

    private double[] MatVec(string name, double[] x, int rows)
    {
        float[] w = _w[name];
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double acc = 0;
            for (int c = 0; c < x.Length; c++) acc += w[r * x.Length + c] * x[c];
            y[r] = acc;
        }
        return y;
    }

    private static double[] RmsNormOnePlus(double[] x, double[] w)
    {
        double ms = x.Sum(e => e * e) / x.Length, scale = 1.0 / Math.Sqrt(ms + Eps);
        return x.Select((e, i) => e * scale * (1.0 + w[i])).ToArray();
    }

    private static double[] L2Norm(double[] x)
    {
        double scale = 1.0 / Math.Sqrt(x.Sum(e => e * e) + 1e-6);
        return x.Select(e => e * scale).ToArray();
    }

    private static double[] Slice(double[] x, int start, int length) => x.AsSpan(start, length).ToArray();

    private static double[][] Add(double[][] a, double[][] b) => a.Select((r, t) => r.Select((e, i) => e + b[t][i]).ToArray()).ToArray();

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double Silu(double x) => x * Sigmoid(x);

    private static double SoftPlus(double x) => x > 20 ? x : Math.Log(1 + Math.Exp(x));
}
