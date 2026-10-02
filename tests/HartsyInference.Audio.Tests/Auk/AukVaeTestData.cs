using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic tiny AuK VAE checkpoint and an independent double-precision reference (naive loops straight from the upstream python) shared by the AukVae tests.</summary>
internal static unsafe class AukVaeTestData
{
    public const int Kernel = 12;

    /// <summary>Hop 6 (2 x 3), latent 4, decoder 8 -> 4 -> 2 channels, two AMP kernels with two conv pairs, two encoder stack layers.</summary>
    public static AukVaeConfig Tiny(bool causal = true) => new()
    {
        UpsampleRates = [3, 2],
        InitialChannels = 8,
        ResblockKernelSizes = [3, 5],
        ResblockDilations = [1, 2],
        LatentDim = 4,
        DownsampleRates = [2, 3],
        DownsampleChannels = [3, 6, 9],
        EncoderStackLayers = 2,
        Causal = causal,
        ActCausal = causal,
    };

    public static Tensor Make(float[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        data.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, data.Length));
        return t;
    }

    public static float[] Values(Tensor t) => new Span<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    public static float[] Random(Random rng, int n, double lo, double hi)
    {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)(lo + rng.NextDouble() * (hi - lo));
        return r;
    }

    public static void AddWeightNormConv(Dictionary<string, Tensor> w, Random rng, string name, int dim0, int dim1, int k, bool bias, double gScale = 1.0)
    {
        w[$"{name}.weight_v"] = Make(Random(rng, dim0 * dim1 * k, -1, 1), dim0, dim1, k);
        w[$"{name}.weight_g"] = Make(Random(rng, dim0, 0.3 * gScale, 0.8 * gScale), dim0, 1, 1);
        if (!bias) return;
        int n = name.Contains("ups.") ? dim1 : dim0;
        w[$"{name}.bias"] = Make(Random(rng, n, -0.1, 0.1), n);
    }

    public static void AddAntiAlias(Dictionary<string, Tensor> w, Random rng, string name, int channels)
    {
        w[$"{name}.act.alpha"] = Make(Random(rng, channels, -0.5, 0.5), channels);
        w[$"{name}.act.beta"] = Make(Random(rng, channels, -0.5, 0.5), channels);
        float[] taps = AntiAliasedSnake.KaiserSincFilter(0.25, 0.3, Kernel);
        float[] up = Random(rng, Kernel, -0.02, 0.02);
        float[] down = Random(rng, Kernel, -0.02, 0.02);
        for (int i = 0; i < Kernel; i++)
        {
            up[i] += taps[i];
            down[i] += taps[i];
        }
        w[$"{name}.upsample.filter"] = Make(up, 1, 1, Kernel);
        w[$"{name}.downsample.lowpass.filter"] = Make(down, 1, 1, Kernel);
    }

    public static Dictionary<string, Tensor> BuildDecoder(AukVaeConfig c, int seed, string prefix = "", double postGain = 1.0)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> w = new();
        AddWeightNormConv(w, rng, $"{prefix}conv_pre", c.InitialChannels, c.LatentDim, 7, true);
        int nk = c.ResblockKernelSizes.Length;
        for (int i = 0; i < c.UpsampleRates.Length; i++)
        {
            int inCh = c.InitialChannels >> i, ch = c.InitialChannels >> (i + 1);
            AddWeightNormConv(w, rng, $"{prefix}ups.{i}.0", inCh, ch, 2 * c.UpsampleRates[i], true);
            for (int j = 0; j < nk; j++)
            {
                string rb = $"{prefix}resblocks.{i * nk + j}";
                for (int p = 0; p < c.ResblockDilations.Length; p++)
                {
                    AddWeightNormConv(w, rng, $"{rb}.convs1.{p}", ch, ch, c.ResblockKernelSizes[j], true, 0.4);
                    AddWeightNormConv(w, rng, $"{rb}.convs2.{p}", ch, ch, c.ResblockKernelSizes[j], true, 0.4);
                }
                for (int a = 0; a < 2 * c.ResblockDilations.Length; a++) AddAntiAlias(w, rng, $"{rb}.activations.{a}", ch);
            }
        }
        AddAntiAlias(w, rng, $"{prefix}activation_post", c.FinalChannels);
        AddWeightNormConv(w, rng, $"{prefix}conv_post", 1, c.FinalChannels, 7, false, postGain);
        return w;
    }

    public static Dictionary<string, Tensor> BuildEncoder(AukVaeConfig c, int seed, string prefix = "")
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> w = new();
        string g = $"{prefix}audio_encoder.generator";
        AddWeightNormConv(w, rng, $"{g}.0.layer", c.DownsampleChannels[0], 1, 3, true);
        int n = c.DownsampleRates.Length;
        for (int b = 0; b < n; b++)
        {
            int idx = 2 + 3 * b, ch = c.DownsampleChannels[b + 1];
            AddWeightNormConv(w, rng, $"{g}.{idx}.layer", ch, c.DownsampleChannels[b], 2 * c.DownsampleRates[b], true);
            for (int i = 0; i < c.EncoderStackLayers; i++)
            {
                AddWeightNormConv(w, rng, $"{g}.{idx + 1}.layers.{i}.1", ch, ch, 3, true, 0.5);
                AddWeightNormConv(w, rng, $"{g}.{idx + 1}.layers.{i}.3", ch, ch, 3, true, 0.5);
            }
        }
        AddWeightNormConv(w, rng, $"{g}.{2 + 3 * n}.layer", 2 * c.LatentDim, c.DownsampleChannels[^1], 3, true);
        return w;
    }

    public static Dictionary<string, Tensor> BuildStats(AukVaeConfig c, int seed, string prefix = "")
    {
        Random rng = new(seed);
        return new Dictionary<string, Tensor>
        {
            [$"{prefix}global_mean"] = Make(Random(rng, c.LatentDim, -1, 1), c.LatentDim),
            [$"{prefix}global_log_std"] = Make(Random(rng, c.LatentDim, 0.5, 2), c.LatentDim),
        };
    }

    // ---------------- independent reference (double precision, [channel][time]) ----------------

    public static double[] Fused(IReadOnlyDictionary<string, Tensor> w, string name)
    {
        Tensor v = w[$"{name}.weight_v"], g = w[$"{name}.weight_g"];
        float[] vv = Values(v), gg = Values(g);
        int rows = (int)v.Shape[0], inner = vv.Length / rows;
        double[] o = new double[vv.Length];
        for (int r = 0; r < rows; r++)
        {
            double norm = 0;
            for (int i = 0; i < inner; i++) norm += (double)vv[r * inner + i] * vv[r * inner + i];
            norm = Math.Sqrt(norm);
            for (int i = 0; i < inner; i++) o[r * inner + i] = gg[r] * vv[r * inner + i] / norm;
        }
        return o;
    }

    public static double[]? Bias(IReadOnlyDictionary<string, Tensor> w, string name) =>
        w.TryGetValue($"{name}.bias", out Tensor? b) ? Array.ConvertAll(Values(b), x => (double)x) : null;

    public static double[][] Conv(double[][] x, double[] w, int co, int ci, int k, double[]? bias, int dil, int padL, int padR, int stride = 1, int groups = 1)
    {
        int t = x[0].Length, outLen = (t + padL + padR - dil * (k - 1) - 1) / stride + 1, icg = ci / groups, ocg = co / groups;
        double[][] o = new double[co][];
        for (int oc = 0; oc < co; oc++)
        {
            o[oc] = new double[outLen];
            for (int ot = 0; ot < outLen; ot++)
            {
                double acc = bias?[oc] ?? 0;
                for (int i = 0; i < icg; i++)
                    for (int kk = 0; kk < k; kk++)
                    {
                        int src = ot * stride + kk * dil - padL;
                        if (src >= 0 && src < t) acc += w[(oc * icg + i) * k + kk] * x[oc / ocg * icg + i][src];
                    }
                o[oc][ot] = acc;
            }
        }
        return o;
    }

    public static double[][] ConvT(double[][] x, double[] w, int ci, int co, int k, double[]? bias, int stride, int cropL, int cropR, int groups = 1)
    {
        int t = x[0].Length, full = (t - 1) * stride + k, icg = ci / groups, ocg = co / groups;
        double[][] o = new double[co][];
        for (int oc = 0; oc < co; oc++)
        {
            double[] row = new double[full];
            int grp = oc / ocg;
            for (int i = 0; i < icg; i++)
                for (int tt = 0; tt < t; tt++)
                    for (int j = 0; j < k; j++) row[tt * stride + j] += x[grp * icg + i][tt] * w[((grp * icg + i) * ocg + oc % ocg) * k + j];
            o[oc] = new double[full - cropL - cropR];
            for (int i = 0; i < o[oc].Length; i++) o[oc][i] = row[cropL + i] + (bias?[oc] ?? 0);
        }
        return o;
    }

    public static double[][] Pad(double[][] x, int l, int r)
    {
        double[][] o = new double[x.Length][];
        for (int c = 0; c < x.Length; c++)
        {
            int t = x[c].Length;
            o[c] = new double[t + l + r];
            for (int i = 0; i < o[c].Length; i++) o[c][i] = x[c][Math.Clamp(i - l, 0, t - 1)];
        }
        return o;
    }

    public static double[][] AntiAlias(IReadOnlyDictionary<string, Tensor> w, string name, double[][] x, bool causalDown)
    {
        int c = x.Length;
        float[] up = Values(w[$"{name}.upsample.filter"]), down = Values(w[$"{name}.downsample.lowpass.filter"]);
        float[] alpha = Values(w[$"{name}.act.alpha"]), beta = Values(w[$"{name}.act.beta"]);
        double[][] padded = Pad(x, 5, 5);
        double[][] u = new double[c][];
        for (int ch = 0; ch < c; ch++)
        {
            double[] full = new double[(padded[ch].Length - 1) * 2 + Kernel];
            for (int t = 0; t < padded[ch].Length; t++)
                for (int j = 0; j < Kernel; j++) full[t * 2 + j] += 2.0 * up[j] * padded[ch][t];
            u[ch] = full[15..(full.Length - 15)];
            double a = Math.Exp(alpha[ch]), b = Math.Exp(beta[ch]);
            for (int t = 0; t < u[ch].Length; t++) u[ch][t] += Math.Pow(Math.Sin(u[ch][t] * a), 2) / (b + 1e-9);
        }
        double[][] dp = causalDown ? Pad(u, 11, 0) : Pad(u, 5, 6);
        double[][] o = new double[c][];
        for (int ch = 0; ch < c; ch++)
        {
            o[ch] = new double[(dp[ch].Length - Kernel) / 2 + 1];
            for (int t = 0; t < o[ch].Length; t++)
                for (int j = 0; j < Kernel; j++) o[ch][t] += down[j] * dp[ch][2 * t + j];
        }
        return o;
    }

    public static double[][] Decode(AukVaeConfig c, IReadOnlyDictionary<string, Tensor> w, double[][] z, string prefix = "")
    {
        double[][] x = Conv(z, Fused(w, $"{prefix}conv_pre"), c.InitialChannels, c.LatentDim, 7, Bias(w, $"{prefix}conv_pre"), 1, 3, 3);
        int nk = c.ResblockKernelSizes.Length;
        for (int i = 0; i < c.UpsampleRates.Length; i++)
        {
            int s = c.UpsampleRates[i], ch = c.InitialChannels >> (i + 1);
            int cropL = c.Causal ? 0 : s / 2, cropR = c.Causal ? s : s / 2;
            x = ConvT(x, Fused(w, $"{prefix}ups.{i}.0"), ch * 2, ch, 2 * s, Bias(w, $"{prefix}ups.{i}.0"), s, cropL, cropR);
            double[][]? sum = null;
            for (int j = 0; j < nk; j++)
            {
                string rb = $"{prefix}resblocks.{i * nk + j}";
                int k = c.ResblockKernelSizes[j];
                double[][] cur = x;
                for (int p = 0; p < c.ResblockDilations.Length; p++)
                {
                    double[][] xt = AntiAlias(w, $"{rb}.activations.{2 * p}", cur, c.ActCausal);
                    int span = c.ResblockDilations[p] * (k - 1);
                    xt = Conv(xt, Fused(w, $"{rb}.convs1.{p}"), ch, ch, k, Bias(w, $"{rb}.convs1.{p}"), c.ResblockDilations[p], c.Causal ? span : span / 2, c.Causal ? 0 : span / 2);
                    xt = AntiAlias(w, $"{rb}.activations.{2 * p + 1}", xt, c.ActCausal);
                    xt = Conv(xt, Fused(w, $"{rb}.convs2.{p}"), ch, ch, k, Bias(w, $"{rb}.convs2.{p}"), 1, c.Causal ? k - 1 : (k - 1) / 2, c.Causal ? 0 : (k - 1) / 2);
                    double[][] next = new double[ch][];
                    for (int cc = 0; cc < ch; cc++) next[cc] = cur[cc].Zip(xt[cc], (a, b) => a + b).ToArray();
                    cur = next;
                }
                sum = sum is null ? cur : sum.Zip(cur, (a, b) => a.Zip(b, (m, n) => m + n).ToArray()).ToArray();
            }
            x = sum!.Select(r => r.Select(v => v / nk).ToArray()).ToArray();
        }
        x = AntiAlias(w, $"{prefix}activation_post", x, c.ActCausal);
        double[][] y = Conv(x, Fused(w, $"{prefix}conv_post"), 1, c.FinalChannels, 7, null, 1, c.Causal ? 6 : 3, c.Causal ? 0 : 3);
        return [y[0].Select(v => Math.Clamp(v, -1.0, 1.0)).ToArray()];
    }

    /// <summary>Encoder mean and log_std stacked as <c>[2D][T]</c>.</summary>
    public static double[][] EncodeStats(AukVaeConfig c, IReadOnlyDictionary<string, Tensor> w, double[] pcm, string prefix = "")
    {
        string g = $"{prefix}audio_encoder.generator";
        double[][] x = Conv([pcm], Fused(w, $"{g}.0.layer"), c.DownsampleChannels[0], 1, 3, Bias(w, $"{g}.0.layer"), 1, 1, 1);
        x = Leaky(x, 0.2);
        for (int b = 0; b < c.DownsampleRates.Length; b++)
        {
            int f = c.DownsampleRates[b], idx = 2 + 3 * b, ch = c.DownsampleChannels[b + 1];
            x = Conv(x, Fused(w, $"{g}.{idx}.layer"), ch, c.DownsampleChannels[b], 2 * f, Bias(w, $"{g}.{idx}.layer"), 1, f - 1, f - 1, f);
            for (int i = 0; i < c.EncoderStackLayers; i++)
            {
                double[][] t = Leaky(x, 0.01);
                t = Conv(t, Fused(w, $"{g}.{idx + 1}.layers.{i}.1"), ch, ch, 3, Bias(w, $"{g}.{idx + 1}.layers.{i}.1"), 1 << i, 1 << i, 1 << i);
                t = Leaky(t, 0.01);
                t = Conv(t, Fused(w, $"{g}.{idx + 1}.layers.{i}.3"), ch, ch, 3, Bias(w, $"{g}.{idx + 1}.layers.{i}.3"), 1, 1, 1);
                x = x.Zip(t, (a, d) => a.Zip(d, (m, n) => m + n).ToArray()).ToArray();
            }
            x = Leaky(x, 0.2);
        }
        string head = $"{g}.{2 + 3 * c.DownsampleRates.Length}.layer";
        return Conv(x, Fused(w, head), 2 * c.LatentDim, c.DownsampleChannels[^1], 3, Bias(w, head), 1, 1, 1);
    }

    private static double[][] Leaky(double[][] x, double slope) => x.Select(r => r.Select(v => v >= 0 ? v : v * slope).ToArray()).ToArray();

    public static double MaxAbsDiff(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        return worst;
    }
}
