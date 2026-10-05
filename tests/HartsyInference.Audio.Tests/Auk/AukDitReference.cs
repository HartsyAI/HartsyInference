using HartsyInference.Audio.Models.Auk;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Synthetic AuK DiT checkpoint plus an independent double-precision naive forward used as the reference for the engine implementation.</summary>
internal sealed class AukDitReference
{
    private const double RmsEps = 1.1920929e-7;
    private const string P = "transformer";

    private readonly AukConfig _cfg;
    private readonly Dictionary<string, (long[] Shape, float[] Data)> _w;

    public AukDitReference(AukConfig cfg, Dictionary<string, (long[] Shape, float[] Data)> weights)
    {
        _cfg = cfg;
        _w = weights;
    }

    public static AukConfig Tiny => new()
    {
        Dim = 32, Heads = 2, DoubleBlocks = 1, SingleBlocks = 1, FfMult = 2, LatentDim = 8, TextDim = 16,
        ConvPosGroups = 2, ConvPosKernel = 3,
    };

    /// <summary>Every checkpoint tensor name the DiT must load, with its shape (mirrors the 420-tensor base.json minus layer_scale/layer_weights).</summary>
    public static Dictionary<string, long[]> KeyShapes(AukConfig c)
    {
        long d = c.Dim, hd = c.HeadDim, inner = c.FfInner;
        Dictionary<string, long[]> s = new();
        void Lin(string k, long o, long i, bool bias = true)
        {
            s[$"{k}.weight"] = [o, i];
            if (bias) s[$"{k}.bias"] = [o];
        }
        Lin($"{P}.time_embed.time_mlp.0", d, c.TimeFreqEmbedDim);
        Lin($"{P}.time_embed.time_mlp.2", d, d);
        Lin($"{P}.audio_embed.linear", d, c.LatentDim);
        foreach (string i in new[] { "0", "2" })
        {
            s[$"{P}.audio_embed.conv_pos_embed.conv1d.{i}.weight"] = [d, d / c.ConvPosGroups, c.ConvPosKernel];
            s[$"{P}.audio_embed.conv_pos_embed.conv1d.{i}.bias"] = [d];
        }
        Lin($"{P}.txt_proj", d, c.TextDim);
        s[$"{P}.txt_norm.weight"] = [d];
        s[$"{P}.rotary_embed.inv_freq"] = [hd / 2];
        for (int b = 0; b < c.DoubleBlocks; b++)
        {
            string k = $"{P}.transformer_blocks.{b}";
            Lin($"{k}.attn_norm_x.linear", 6 * d, d);
            Lin($"{k}.attn_norm_c.linear", 6 * d, d);
            Lin($"{k}.attn.to_qkv", 3 * d, d);
            Lin($"{k}.attn.to_qkv_c", 3 * d, d);
            Lin($"{k}.attn.to_out.0", d, d);
            Lin($"{k}.attn.to_out_c", d, d);
            foreach (string n in new[] { "q_norm", "k_norm", "c_q_norm", "c_k_norm" }) s[$"{k}.attn.{n}.weight"] = [hd];
            foreach (string f in new[] { "ff_x", "ff_c" })
            {
                Lin($"{k}.{f}.linear_in", 2 * inner, d, false);
                Lin($"{k}.{f}.linear_out", d, inner, false);
            }
        }
        for (int b = 0; b < c.SingleBlocks; b++)
        {
            string k = $"{P}.single_transformer_blocks.{b}";
            Lin($"{k}.attn_norm.linear", 6 * d, d);
            Lin($"{k}.attn.to_qkv", 3 * d, d);
            Lin($"{k}.attn.to_out.0", d, d);
            s[$"{k}.attn.q_norm.weight"] = [hd];
            s[$"{k}.attn.k_norm.weight"] = [hd];
            Lin($"{k}.ff.linear_in", 2 * inner, d, false);
            Lin($"{k}.ff.linear_out", d, inner, false);
        }
        Lin($"{P}.norm_out.linear", 2 * d, d);
        Lin($"{P}.proj_out", c.LatentDim, d);
        return s;
    }

    public static Dictionary<string, (long[] Shape, float[] Data)> RandomWeights(AukConfig c, int seed, float[]? invFreq = null)
    {
        Random rng = new(seed);
        Dictionary<string, (long[] Shape, float[] Data)> w = new();
        foreach ((string key, long[] shape) in KeyShapes(c))
        {
            long count = 1;
            foreach (long dim in shape) count *= dim;
            float[] data = new float[count];
            bool norm = key.Contains("_norm.weight") || key.Contains("txt_norm");
            for (int i = 0; i < data.Length; i++)
            {
                double g = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
                data[i] = norm ? (float)(1 + 0.1 * g) : (float)(0.15 * g);
            }
            w[key] = (shape, data);
        }
        w[$"{P}.rotary_embed.inv_freq"] = (w[$"{P}.rotary_embed.inv_freq"].Shape, invFreq ?? AukRope.TheoryInvFreq(c.HeadDim));
        return w;
    }

    public static unsafe Tensor ToTensor(long[] shape, float[] data)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        data.AsSpan().CopyTo(new Span<float>((float*)t.DataPointer, data.Length));
        return t;
    }

    public static Dictionary<string, Tensor> ToTensors(Dictionary<string, (long[] Shape, float[] Data)> w)
    {
        Dictionary<string, Tensor> r = new();
        foreach ((string k, (long[] shape, float[] data)) in w) r[k] = ToTensor(shape, data);
        return r;
    }

    /// <summary>Naive forward in double precision; <paramref name="refl"/> null means no reference. Returns <c>[N][latent]</c>.</summary>
    public double[][] Forward(double[][] noisy, double[][]? refl, double[][] text, double t, bool dropText, bool dropAudio)
    {
        int dim = _cfg.Dim;
        double[] st = Silu(TimeEmbed(t));
        double[][] c = dropText ? Zeros(text.Length, dim) : text.Select(r => Rms(Lin(r, $"{P}.txt_proj"), $"{P}.txt_norm.weight")).ToArray();
        double[][] x = Embed(noisy);
        int r0 = 0;
        if (refl is not null)
        {
            r0 = refl.Length;
            double[][] rin = dropAudio ? Zeros(refl.Length, _cfg.LatentDim) : refl;
            x = Embed(rin).Concat(x).ToArray();
        }
        int nt = c.Length;
        for (int b = 0; b < _cfg.DoubleBlocks; b++) (c, x) = Double(b, x, c, st);
        double[][] all = c.Concat(x).ToArray();
        for (int b = 0; b < _cfg.SingleBlocks; b++) all = Single(b, all, st);
        double[][] tail = all.Skip(nt + r0).ToArray();
        double[] mods = Lin(st, $"{P}.norm_out.linear");
        double[][] outRows = new double[tail.Length][];
        for (int i = 0; i < tail.Length; i++)
        {
            double[] ln = Ln(tail[i]);
            double[] h = new double[dim];
            for (int d = 0; d < dim; d++) h[d] = ln[d] * (1 + mods[d]) + mods[dim + d];
            outRows[i] = Lin(h, $"{P}.proj_out");
        }
        return outRows;
    }

    private double[][] Embed(double[][] x)
    {
        int dim = _cfg.Dim;
        double[][] lin = x.Select(r => Lin(r, $"{P}.audio_embed.linear")).ToArray();
        double[][] a = Conv(lin, $"{P}.audio_embed.conv_pos_embed.conv1d.0");
        a = a.Select(r => r.Select(Mish).ToArray()).ToArray();
        a = Conv(a, $"{P}.audio_embed.conv_pos_embed.conv1d.2");
        a = a.Select(r => r.Select(Mish).ToArray()).ToArray();
        double[][] res = new double[x.Length][];
        for (int i = 0; i < x.Length; i++)
        {
            res[i] = new double[dim];
            for (int d = 0; d < dim; d++) res[i][d] = lin[i][d] + a[i][d];
        }
        return res;
    }

    private double[][] Conv(double[][] x, string key)
    {
        int dim = _cfg.Dim, g = _cfg.ConvPosGroups, k = _cfg.ConvPosKernel, cpg = dim / g, pad = k / 2;
        float[] wt = _w[$"{key}.weight"].Data, bs = _w[$"{key}.bias"].Data;
        double[][] o = new double[x.Length][];
        for (int t = 0; t < x.Length; t++)
        {
            o[t] = new double[dim];
            for (int ch = 0; ch < dim; ch++)
            {
                double s = bs[ch];
                int grp = ch / cpg;
                for (int ci = 0; ci < cpg; ci++)
                    for (int kk = 0; kk < k; kk++)
                    {
                        int tt = t + kk - pad;
                        if (tt >= 0 && tt < x.Length) s += wt[(ch * cpg + ci) * k + kk] * x[tt][grp * cpg + ci];
                    }
                o[t][ch] = s;
            }
        }
        return o;
    }

    private double[] TimeEmbed(double t)
    {
        int f = _cfg.TimeFreqEmbedDim, half = f / 2;
        double factor = Math.Log(10000.0) / (half - 1);
        double[] e = new double[f];
        for (int i = 0; i < half; i++)
        {
            double a = 1000 * t * Math.Exp(-factor * i);
            e[i] = Math.Sin(a);
            e[half + i] = Math.Cos(a);
        }
        return Lin(Silu(Lin(e, $"{P}.time_embed.time_mlp.0")), $"{P}.time_embed.time_mlp.2");
    }

    private (double[][] C, double[][] X) Double(int b, double[][] x, double[][] c, double[] st)
    {
        string k = $"{P}.transformer_blocks.{b}";
        double[][] mx = Mods(st, $"{k}.attn_norm_x.linear"), mc = Mods(st, $"{k}.attn_norm_c.linear");
        double[][] nx = x.Select(r => Mod(r, mx[1], mx[0])).ToArray();
        double[][] nc = c.Select(r => Mod(r, mc[1], mc[0])).ToArray();
        (double[][] qx, double[][] kx, double[][] vx) = Qkv(nx, $"{k}.attn.to_qkv", $"{k}.attn.q_norm.weight", $"{k}.attn.k_norm.weight");
        (double[][] qc, double[][] kc, double[][] vc) = Qkv(nc, $"{k}.attn.to_qkv_c", $"{k}.attn.c_q_norm.weight", $"{k}.attn.c_k_norm.weight");
        double[][] att = Attention(qx.Concat(qc).ToArray(), kx.Concat(kc).ToArray(), vx.Concat(vc).ToArray());
        double[][] ax = att.Take(x.Length).Select(r => Lin(r, $"{k}.attn.to_out.0")).ToArray();
        double[][] ac = att.Skip(x.Length).Select(r => Lin(r, $"{k}.attn.to_out_c")).ToArray();
        double[][] cn = Gated(c, ac, mc[2]);
        double[][] xn = Gated(x, ax, mx[2]);
        cn = Ffn(cn, mc, $"{k}.ff_c");
        xn = Ffn(xn, mx, $"{k}.ff_x");
        return (cn, xn);
    }

    private double[][] Single(int b, double[][] x, double[] st)
    {
        string k = $"{P}.single_transformer_blocks.{b}";
        double[][] m = Mods(st, $"{k}.attn_norm.linear");
        double[][] n = x.Select(r => Mod(r, m[1], m[0])).ToArray();
        (double[][] q, double[][] kk, double[][] v) = Qkv(n, $"{k}.attn.to_qkv", $"{k}.attn.q_norm.weight", $"{k}.attn.k_norm.weight");
        double[][] a = Attention(q, kk, v).Select(r => Lin(r, $"{k}.attn.to_out.0")).ToArray();
        return Ffn(Gated(x, a, m[2]), m, $"{k}.ff");
    }

    private double[][] Ffn(double[][] x, double[][] m, string key)
    {
        int inner = _cfg.FfInner;
        double[][] n = x.Select(r => Mod(r, m[4], m[3])).ToArray();
        double[][] o = new double[x.Length][];
        for (int i = 0; i < x.Length; i++)
        {
            double[] h = Lin(n[i], $"{key}.linear_in", false);
            double[] a = new double[inner];
            for (int j = 0; j < inner; j++) a[j] = Silu(h[j]) * h[inner + j];
            double[] f = Lin(a, $"{key}.linear_out", false);
            o[i] = new double[x[i].Length];
            for (int d = 0; d < f.Length; d++) o[i][d] = x[i][d] + m[5][d] * f[d];
        }
        return o;
    }

    private double[][] Gated(double[][] x, double[][] v, double[] gate)
    {
        double[][] o = new double[x.Length][];
        for (int i = 0; i < x.Length; i++)
        {
            o[i] = new double[x[i].Length];
            for (int d = 0; d < o[i].Length; d++) o[i][d] = x[i][d] + gate[d] * v[i][d];
        }
        return o;
    }

    private (double[][] Q, double[][] K, double[][] V) Qkv(double[][] x, string qkv, string qn, string kn)
    {
        int dim = _cfg.Dim, hd = _cfg.HeadDim;
        double[][] q = new double[x.Length][], k = new double[x.Length][], v = new double[x.Length][];
        float[] inv = _w[$"{P}.rotary_embed.inv_freq"].Data;
        for (int p = 0; p < x.Length; p++)
        {
            double[] f = Lin(x[p], qkv);
            q[p] = f[..dim]; k[p] = f[dim..(2 * dim)]; v[p] = f[(2 * dim)..];
            for (int h = 0; h < _cfg.Heads; h++)
            {
                Norm(q[p], h * hd, hd, qn);
                Norm(k[p], h * hd, hd, kn);
                Rope(q[p], h * hd, hd, p, inv);
                Rope(k[p], h * hd, hd, p, inv);
            }
        }
        return (q, k, v);
    }

    private void Norm(double[] v, int off, int n, string wk)
    {
        float[] w = _w[wk].Data;
        double ms = 0;
        for (int i = 0; i < n; i++) ms += v[off + i] * v[off + i];
        double s = 1 / Math.Sqrt(ms / n + RmsEps);
        for (int i = 0; i < n; i++) v[off + i] = v[off + i] * s * w[i];
    }

    private static void Rope(double[] v, int off, int n, int pos, float[] inv)
    {
        for (int i = 0; i < n / 2; i++)
        {
            double a = pos * (double)inv[i], cs = Math.Cos(a), sn = Math.Sin(a);
            double x0 = v[off + 2 * i], x1 = v[off + 2 * i + 1];
            v[off + 2 * i] = x0 * cs - x1 * sn;
            v[off + 2 * i + 1] = x1 * cs + x0 * sn;
        }
    }

    private double[][] Attention(double[][] q, double[][] k, double[][] v)
    {
        int hd = _cfg.HeadDim, n = q.Length;
        double[][] o = new double[n][];
        for (int i = 0; i < n; i++) o[i] = new double[_cfg.Dim];
        for (int h = 0; h < _cfg.Heads; h++)
            for (int i = 0; i < n; i++)
            {
                double[] sc = new double[n];
                double mx = double.MinValue;
                for (int j = 0; j < n; j++)
                {
                    double s = 0;
                    for (int d = 0; d < hd; d++) s += q[i][h * hd + d] * k[j][h * hd + d];
                    sc[j] = s / Math.Sqrt(hd);
                    mx = Math.Max(mx, sc[j]);
                }
                double sum = 0;
                for (int j = 0; j < n; j++) { sc[j] = Math.Exp(sc[j] - mx); sum += sc[j]; }
                for (int j = 0; j < n; j++)
                    for (int d = 0; d < hd; d++) o[i][h * hd + d] += sc[j] / sum * v[j][h * hd + d];
            }
        return o;
    }

    private double[][] Mods(double[] st, string key)
    {
        double[] m = Lin(st, key);
        int d = _cfg.Dim;
        double[][] r = new double[6][];
        for (int i = 0; i < 6; i++) r[i] = m[(i * d)..((i + 1) * d)];
        for (int j = 0; j < d; j++) { r[1][j] += 1; r[4][j] += 1; }
        return r;
    }

    private double[] Mod(double[] x, double[] scale, double[] shift)
    {
        double[] ln = Ln(x);
        for (int i = 0; i < ln.Length; i++) ln[i] = ln[i] * scale[i] + shift[i];
        return ln;
    }

    private static double[] Ln(double[] x)
    {
        double mean = x.Average();
        double var = x.Sum(v => (v - mean) * (v - mean)) / x.Length;
        return x.Select(v => (v - mean) / Math.Sqrt(var + 1e-6)).ToArray();
    }

    private double[] Rms(double[] x, string wk)
    {
        float[] w = _w[wk].Data;
        double ms = x.Sum(v => v * v) / x.Length;
        double s = 1 / Math.Sqrt(ms + RmsEps);
        return x.Select((v, i) => v * s * w[i]).ToArray();
    }

    private double[] Lin(double[] x, string key, bool bias = true)
    {
        (long[] shape, float[] data) = _w[$"{key}.weight"];
        int o = (int)shape[0], n = (int)shape[1];
        float[]? b = bias ? _w[$"{key}.bias"].Data : null;
        double[] r = new double[o];
        for (int i = 0; i < o; i++)
        {
            double s = b?[i] ?? 0;
            for (int j = 0; j < n; j++) s += data[i * n + j] * x[j];
            r[i] = s;
        }
        return r;
    }

    private static double[][] Zeros(int n, int d) => Enumerable.Range(0, n).Select(_ => new double[d]).ToArray();

    private static double Silu(double x) => x / (1 + Math.Exp(-x));

    private static double[] Silu(double[] x) => x.Select(Silu).ToArray();

    private static double Mish(double x) => x * Math.Tanh(Math.Log(1 + Math.Exp(x)));
}
