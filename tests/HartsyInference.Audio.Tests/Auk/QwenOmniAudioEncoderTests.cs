using Xunit;
using HartsyInference.Audio.Models.QwenOmni;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Structural tests of the Qwen2.5-Omni audio tower on a tiny config against a double-precision naive reference; real-weight parity is validation-pending.</summary>
public sealed unsafe class QwenOmniAudioEncoderTests
{
    private const string Prefix = "thinker.audio_tower";

    private static QwenOmniConfig Tiny => new()
    {
        NumMelBins = 4, DModel = 8, Layers = 2, Heads = 2, FfnDim = 16, NWindow = 4, OutputDim = 6,
    };

    [Fact]
    public void ExpectedWeights_MatchTheCheckpointKeySetForTheRealConfig()
    {
        QwenOmniAudioEncoder encoder = new(QwenOmniConfig.Default);
        List<string> keys = [.. encoder.ExpectedWeights().Select(e => e.Key)];
        Assert.Equal(4 + 32 * 15 + 4, keys.Count);
        Assert.Contains("layers.31.self_attn.k_proj.weight", keys);
        Assert.DoesNotContain("layers.0.self_attn.k_proj.bias", keys);
        Assert.Contains("proj.weight", keys);
        (string _, long[] shape) = encoder.ExpectedWeights().First(e => e.Key == "proj.weight");
        Assert.Equal([2048L, 1280L], shape);
    }

    [Fact]
    public void LoadWeights_MissingKey_FailsFastNamingIt()
    {
        QwenOmniAudioEncoder encoder = new(Tiny);
        Dictionary<string, Tensor> w = Weights(Tiny, 1);
        w.Remove($"{Prefix}.layers.1.fc2.bias");
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => encoder.LoadWeights(w));
        Assert.Contains("layers.1.fc2.bias", ex.Message);
    }

    [Fact]
    public void LoadWeights_WrongShape_FailsFast()
    {
        QwenOmniAudioEncoder encoder = new(Tiny);
        Dictionary<string, Tensor> w = Weights(Tiny, 1);
        w[$"{Prefix}.proj.weight"] = Rand([5, 8], new Random(3), 0.3);
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => encoder.LoadWeights(w));
        Assert.Contains("proj.weight", ex.Message);
    }

    [Fact]
    public void Forward_BeforeLoad_OrWithTooFewFrames_Throws()
    {
        CpuBackend backend = new();
        QwenOmniAudioEncoder encoder = new(Tiny);
        Tensor mel = Rand([4, 20], new Random(2), 1.0);
        Assert.Throws<InvalidOperationException>(() => encoder.Forward(backend, mel, 20));
        encoder.LoadWeights(Weights(Tiny, 1));
        Assert.Throws<ArgumentException>(() => encoder.Forward(backend, mel, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => encoder.Forward(backend, mel, 21));
    }

    [Fact]
    public void Sinusoids_MatchTheClosedForm()
    {
        const int length = 7;
        const int channels = 10;
        float[] table = new float[length * channels];
        QwenOmniAudioEncoder.FillSinusoids(table, length, channels);
        for (int t = 0; t < length; t++)
        {
            for (int i = 0; i < channels / 2; i++)
            {
                double inv = Math.Pow(10_000.0, -(double)i / (channels / 2 - 1));
                Assert.True(Math.Abs(Math.Sin(t * inv) - table[t * channels + i]) < 1e-5);
                Assert.True(Math.Abs(Math.Cos(t * inv) - table[t * channels + channels / 2 + i]) < 1e-5);
            }
        }
        Assert.Equal(0f, table[0]);
        Assert.Equal(1f, table[channels / 2]);
        Assert.True(Math.Abs(Math.Sin(1.0) - table[channels]) < 1e-5);
        Assert.True(Math.Abs(Math.Cos(Math.Pow(10_000.0, -0.25)) - table[channels + channels / 2 + 1]) < 1e-5);
    }

    [Theory]
    [InlineData(16)]
    [InlineData(21)]
    [InlineData(8)]
    [InlineData(5)]
    [InlineData(3)]
    public void Forward_MatchesNaiveDoubleReference(int frames)
    {
        QwenOmniConfig cfg = Tiny;
        CpuBackend backend = new();
        Dictionary<string, Tensor> w = Weights(cfg, 7);
        QwenOmniAudioEncoder encoder = new(cfg);
        encoder.LoadWeights(w);
        Tensor mel = Rand([cfg.NumMelBins, frames + 3], new Random(frames), 1.0);

        Tensor output = encoder.Forward(backend, mel, frames);

        double[][] expected = Reference(cfg, w, mel, frames);
        Assert.Equal(QwenOmniProcessor.AudioTokens(frames), (int)output.Shape[0]);
        Assert.Equal(cfg.OutputDim, (int)output.Shape[1]);
        Assert.Equal(expected.Length, (int)output.Shape[0]);
        float* o = (float*)output.DataPointer;
        for (int r = 0; r < expected.Length; r++)
        {
            for (int c = 0; c < cfg.OutputDim; c++) Assert.True(Math.Abs(expected[r][c] - o[r * cfg.OutputDim + c]) < 2e-4, $"[{r},{c}] {expected[r][c]} vs {o[r * cfg.OutputDim + c]}");
        }
        output.Dispose();
    }

    [Fact]
    public void Attention_IsBlockDiagonalAcrossChunks()
    {
        QwenOmniConfig cfg = Tiny;
        CpuBackend backend = new();
        QwenOmniAudioEncoder encoder = new(cfg);
        encoder.LoadWeights(Weights(cfg, 11));
        const int frames = 24;
        Tensor a = Rand([cfg.NumMelBins, frames], new Random(5), 1.0);
        Tensor b = Rand([cfg.NumMelBins, frames], new Random(6), 1.0);
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        for (int m = 0; m < cfg.NumMelBins; m++)
        {
            for (int t = 0; t < 8; t++) pb[m * frames + t] = pa[m * frames + t];
        }

        Tensor outA = encoder.Forward(backend, a, frames);
        Tensor outB = encoder.Forward(backend, b, frames);

        float* oa = (float*)outA.DataPointer;
        float* ob = (float*)outB.DataPointer;
        int firstChunkTokens = 2;
        for (int i = 0; i < firstChunkTokens * cfg.OutputDim; i++) Assert.Equal(oa[i], ob[i]);
        double later = 0;
        for (int i = firstChunkTokens * cfg.OutputDim; i < outA.Shape.ElementCount; i++) later += Math.Abs(oa[i] - ob[i]);
        Assert.True(later > 1e-3, "later chunks must depend on their own mel");
    }

    private static double Erf(double x)
    {
        double sum = 0;
        double term = x;
        for (int n = 0; n < 120; n++)
        {
            sum += term / (2 * n + 1);
            term *= -x * x / (n + 1);
        }
        return 2.0 / Math.Sqrt(Math.PI) * sum;
    }

    private static double Gelu(double x) => 0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0)));

    private static double[] W(Dictionary<string, Tensor> w, string key)
    {
        Tensor t = w[$"{Prefix}.{key}"];
        float* p = (float*)t.DataPointer;
        double[] r = new double[t.Shape.ElementCount];
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    private static double[][] LayerNorm(double[][] x, double[] g, double[] b)
    {
        double[][] y = new double[x.Length][];
        for (int t = 0; t < x.Length; t++)
        {
            double mean = x[t].Average();
            double var = x[t].Select(v => (v - mean) * (v - mean)).Average();
            y[t] = new double[x[t].Length];
            for (int i = 0; i < y[t].Length; i++) y[t][i] = (x[t][i] - mean) / Math.Sqrt(var + 1e-5) * g[i] + b[i];
        }
        return y;
    }

    private static double[][] Linear(double[][] x, double[] weight, double[]? bias, int outDim)
    {
        int inDim = x[0].Length;
        double[][] y = new double[x.Length][];
        for (int t = 0; t < x.Length; t++)
        {
            y[t] = new double[outDim];
            for (int o = 0; o < outDim; o++)
            {
                double s = bias?[o] ?? 0;
                for (int i = 0; i < inDim; i++) s += weight[o * inDim + i] * x[t][i];
                y[t][o] = s;
            }
        }
        return y;
    }

    private static double[][] Layer(QwenOmniConfig cfg, Dictionary<string, Tensor> w, int l, double[][] h)
    {
        string p = $"layers.{l}";
        int d = cfg.DModel;
        int hd = cfg.HeadDim;
        double[][] n1 = LayerNorm(h, W(w, $"{p}.self_attn_layer_norm.weight"), W(w, $"{p}.self_attn_layer_norm.bias"));
        double[][] q = Linear(n1, W(w, $"{p}.self_attn.q_proj.weight"), W(w, $"{p}.self_attn.q_proj.bias"), d);
        double[][] k = Linear(n1, W(w, $"{p}.self_attn.k_proj.weight"), null, d);
        double[][] v = Linear(n1, W(w, $"{p}.self_attn.v_proj.weight"), W(w, $"{p}.self_attn.v_proj.bias"), d);
        double[][] ctx = new double[h.Length][];
        for (int t = 0; t < h.Length; t++) ctx[t] = new double[d];
        for (int head = 0; head < cfg.Heads; head++)
        {
            for (int t = 0; t < h.Length; t++)
            {
                double[] score = new double[h.Length];
                for (int u = 0; u < h.Length; u++)
                {
                    for (int i = 0; i < hd; i++) score[u] += q[t][head * hd + i] * k[u][head * hd + i];
                    score[u] /= Math.Sqrt(hd);
                }
                double max = score.Max();
                double[] e = score.Select(s => Math.Exp(s - max)).ToArray();
                double z = e.Sum();
                for (int u = 0; u < h.Length; u++)
                {
                    for (int i = 0; i < hd; i++) ctx[t][head * hd + i] += e[u] / z * v[u][head * hd + i];
                }
            }
        }
        double[][] o = Linear(ctx, W(w, $"{p}.self_attn.out_proj.weight"), W(w, $"{p}.self_attn.out_proj.bias"), d);
        double[][] r1 = new double[h.Length][];
        for (int t = 0; t < h.Length; t++) r1[t] = h[t].Zip(o[t], (a, b) => a + b).ToArray();
        double[][] n2 = LayerNorm(r1, W(w, $"{p}.final_layer_norm.weight"), W(w, $"{p}.final_layer_norm.bias"));
        double[][] f = Linear(n2, W(w, $"{p}.fc1.weight"), W(w, $"{p}.fc1.bias"), cfg.FfnDim);
        foreach (double[] row in f) for (int i = 0; i < row.Length; i++) row[i] = Gelu(row[i]);
        double[][] f2 = Linear(f, W(w, $"{p}.fc2.weight"), W(w, $"{p}.fc2.bias"), d);
        double[][] r2 = new double[h.Length][];
        for (int t = 0; t < h.Length; t++) r2[t] = r1[t].Zip(f2[t], (a, b) => a + b).ToArray();
        return r2;
    }

    private static double[][] Reference(QwenOmniConfig cfg, Dictionary<string, Tensor> w, Tensor mel, int frames)
    {
        int d = cfg.DModel;
        int bins = cfg.NumMelBins;
        int stride = (int)mel.Shape[1];
        float* mp = (float*)mel.DataPointer;
        double[] c1w = W(w, "conv1.weight");
        double[] c1b = W(w, "conv1.bias");
        double[] c2w = W(w, "conv2.weight");
        double[] c2b = W(w, "conv2.bias");
        List<double[]> all = [];
        for (int start = 0; start < frames; start += cfg.ChunkFrames)
        {
            int len = Math.Min(cfg.ChunkFrames, frames - start);
            double[][] g1 = new double[d][];
            for (int c = 0; c < d; c++)
            {
                g1[c] = new double[len];
                for (int t = 0; t < len; t++)
                {
                    double s = c1b[c];
                    for (int m = 0; m < bins; m++)
                    {
                        for (int kk = 0; kk < 3; kk++)
                        {
                            int src = t + kk - 1;
                            if (src >= 0 && src < len) s += c1w[(c * bins + m) * 3 + kk] * mp[m * stride + start + src];
                        }
                    }
                    g1[c][t] = Gelu(s);
                }
            }
            int outLen = (len - 1) / 2 + 1;
            double[][] seq = new double[outLen][];
            float[] pos = new float[cfg.NWindow * d];
            QwenOmniAudioEncoder.FillSinusoids(pos, cfg.NWindow, d);
            for (int j = 0; j < outLen; j++)
            {
                seq[j] = new double[d];
                for (int c = 0; c < d; c++)
                {
                    double s = c2b[c];
                    for (int m = 0; m < d; m++)
                    {
                        for (int kk = 0; kk < 3; kk++)
                        {
                            int src = 2 * j + kk - 1;
                            if (src >= 0 && src < len) s += c2w[(c * d + m) * 3 + kk] * g1[m][src];
                        }
                    }
                    seq[j][c] = Gelu(s) + pos[j * d + c];
                }
            }
            for (int l = 0; l < cfg.Layers; l++) seq = Layer(cfg, w, l, seq);
            all.AddRange(seq);
        }
        int pooled = all.Count / 2;
        double[][] avg = new double[pooled][];
        for (int j = 0; j < pooled; j++) avg[j] = all[2 * j].Zip(all[2 * j + 1], (a, b) => 0.5 * (a + b)).ToArray();
        double[][] normed = LayerNorm(avg, W(w, "ln_post.weight"), W(w, "ln_post.bias"));
        return Linear(normed, W(w, "proj.weight"), W(w, "proj.bias"), cfg.OutputDim);
    }

    private static Dictionary<string, Tensor> Weights(QwenOmniConfig cfg, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> w = [];
        foreach ((string key, long[] shape) in new QwenOmniAudioEncoder(cfg).ExpectedWeights())
        {
            Tensor t = Rand(shape, rng, key.EndsWith("norm.weight", StringComparison.Ordinal) || key == "ln_post.weight" ? 0.1 : 0.4);
            if (key.EndsWith("norm.weight", StringComparison.Ordinal) || key == "ln_post.weight")
            {
                float* p = (float*)t.DataPointer;
                for (long i = 0; i < t.Shape.ElementCount; i++) p[i] += 1f;
            }
            w[$"{Prefix}.{key}"] = t;
        }
        return w;
    }

    private static Tensor Rand(long[] shape, Random rng, double amplitude)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * amplitude);
        return t;
    }
}
