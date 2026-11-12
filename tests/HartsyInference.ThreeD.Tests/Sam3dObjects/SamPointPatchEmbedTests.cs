using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamPointPatchEmbedTests
{
    // 8x8 input in 4x4 windows (four windows), width 8, two heads, MLP hidden 16, sinh remap so the remap is not a no-op.
    private static readonly SamPointPatchConfig Config = new(InputSize: 8, PatchSize: 4, EmbedDim: 8, Heads: 2, MlpRatio: 2.0, Remap: SamPointRemapType.Sinh);

    private sealed class Owned(IEnumerable<Tensor> tensors) : IDisposable
    {
        public void Dispose()
        {
            foreach (Tensor t in tensors) t.Dispose();
        }
    }

    private static Tensor Filled(Random rng, double scale, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>())
            v = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static Dictionary<string, Tensor> Weights(Random rng)
    {
        int d = Config.EmbedDim, side = Config.WindowsPerSide, hidden = Config.MlpHidden;
        return new Dictionary<string, Tensor>
        {
            ["point_proj.weight"] = Filled(rng, 1.0, d, 3),
            ["point_proj.bias"] = Filled(rng, 0.5, d),
            ["invalid_xyz_token"] = Filled(rng, 1.0, d),
            ["pos_embed"] = Filled(rng, 1.0, 1, d, side, side),
            ["pos_embed_window"] = Filled(rng, 1.0, 1, Config.WindowTokens, d),
            ["cls_token"] = Filled(rng, 1.0, 1, 1, d),
            ["blocks.0.norm1.weight"] = Filled(rng, 1.0, d),
            ["blocks.0.norm1.bias"] = Filled(rng, 0.4, d),
            ["blocks.0.attn.qkv.weight"] = Filled(rng, 1.0, 3 * d, d),
            ["blocks.0.attn.qkv.bias"] = Filled(rng, 0.4, 3 * d),
            ["blocks.0.attn.proj.weight"] = Filled(rng, 1.0, d, d),
            ["blocks.0.attn.proj.bias"] = Filled(rng, 0.4, d),
            ["blocks.0.norm2.weight"] = Filled(rng, 1.0, d),
            ["blocks.0.norm2.bias"] = Filled(rng, 0.4, d),
            ["blocks.0.mlp.fc1.weight"] = Filled(rng, 1.0, hidden, d),
            ["blocks.0.mlp.fc1.bias"] = Filled(rng, 0.4, hidden),
            ["blocks.0.mlp.fc2.weight"] = Filled(rng, 1.0, d, hidden),
            ["blocks.0.mlp.fc2.bias"] = Filled(rng, 0.4, d),
        };
    }

    private static float[] V(Dictionary<string, Tensor> w, string key) => w[key].AsReadOnlySpan<float>().ToArray();

    private static double Erf(double x)
    {
        // Abramowitz-Stegun 7.1.26 is too coarse for a 1e-3 check on larger values, so integrate the series instead.
        double sum = x, term = x;
        for (int n = 1; n < 60; n++)
        {
            term *= -x * x / n;
            sum += term / (2 * n + 1);
        }

        return 2 / Math.Sqrt(Math.PI) * sum;
    }

    private static double[] Linear(double[] x, float[] w, float[] b)
    {
        int outDim = b.Length, inDim = w.Length / outDim;
        double[] y = new double[outDim];
        for (int o = 0; o < outDim; o++)
        {
            double s = b[o];
            for (int k = 0; k < inDim; k++) s += w[o * inDim + k] * x[k];
            y[o] = s;
        }

        return y;
    }

    private static double[] LayerNorm(double[] x, float[] w, float[] b)
    {
        double mean = x.Average(), var = x.Sum(v => (v - mean) * (v - mean)) / x.Length;
        return x.Select((v, i) => (v - mean) / Math.Sqrt(var + 1e-6) * w[i] + b[i]).ToArray();
    }

    /// <summary>The upstream forward in doubles for one pointmap given as planar [3, h, w].</summary>
    private static double[][] Reference(Dictionary<string, Tensor> w, float[] map, int h, int wd)
    {
        int size = Config.InputSize, patch = Config.PatchSize, side = Config.WindowsPerSide, d = Config.EmbedDim, heads = Config.Heads, hd = d / heads;
        // 1. nearest resize, validity, zero invalid, sinh remap, projection (invalid points take the token only).
        double[][][] x = new double[size][][];
        for (int y = 0; y < size; y++)
        {
            x[y] = new double[size][];
            for (int xx = 0; xx < size; xx++)
            {
                int sy = y * h / size, sx = xx * wd / size;
                double[] p = [map[0 * h * wd + sy * wd + sx], map[1 * h * wd + sy * wd + sx], map[2 * h * wd + sy * wd + sx]];
                bool valid = p.All(double.IsFinite);
                if (!valid)
                {
                    x[y][xx] = V(w, "invalid_xyz_token").Select(v => (double)v).ToArray();
                    continue;
                }

                x[y][xx] = Linear(p.Select(Math.Asinh).ToArray(), V(w, "point_proj.weight"), V(w, "point_proj.bias"));
            }
        }

        float[] cls = V(w, "cls_token"), posWindow = V(w, "pos_embed_window"), posPatch = V(w, "pos_embed");
        double[][] outputs = new double[side * side][];
        for (int wy = 0; wy < side; wy++)
        {
            for (int wx = 0; wx < side; wx++)
            {
                // 2. window tokens: cls first, then the pixels row-major, plus the in-window position.
                List<double[]> tokens = [cls.Select(v => (double)v).ToArray()];
                for (int py = 0; py < patch; py++)
                {
                    for (int px = 0; px < patch; px++)
                        tokens.Add(x[wy * patch + py][wx * patch + px]);
                }

                double[][] t = tokens.Select((row, i) => row.Select((v, c) => v + posWindow[i * d + c]).ToArray()).ToArray();

                // 3. timm Block.
                double[][] n1 = t.Select(r => LayerNorm(r, V(w, "blocks.0.norm1.weight"), V(w, "blocks.0.norm1.bias"))).ToArray();
                double[][] qkv = n1.Select(r => Linear(r, V(w, "blocks.0.attn.qkv.weight"), V(w, "blocks.0.attn.qkv.bias"))).ToArray();
                double[][] att = t.Select(_ => new double[d]).ToArray();
                for (int hh = 0; hh < heads; hh++)
                {
                    for (int i = 0; i < t.Length; i++)
                    {
                        double[] s = new double[t.Length];
                        for (int j = 0; j < t.Length; j++)
                        {
                            double dot = 0;
                            for (int c = 0; c < hd; c++) dot += qkv[i][hh * hd + c] * qkv[j][d + hh * hd + c];
                            s[j] = dot / Math.Sqrt(hd);
                        }

                        double max = s.Max(), sum = s.Sum(v => Math.Exp(v - max));
                        for (int j = 0; j < t.Length; j++)
                        {
                            for (int c = 0; c < hd; c++) att[i][hh * hd + c] += Math.Exp(s[j] - max) / sum * qkv[j][2 * d + hh * hd + c];
                        }
                    }
                }

                double[][] x1 = t.Select((r, i) =>
                {
                    double[] o = Linear(att[i], V(w, "blocks.0.attn.proj.weight"), V(w, "blocks.0.attn.proj.bias"));
                    return r.Select((v, c) => v + o[c]).ToArray();
                }).ToArray();
                double[] first = x1[0];
                double[] n2 = LayerNorm(first, V(w, "blocks.0.norm2.weight"), V(w, "blocks.0.norm2.bias"));
                double[] up = Linear(n2, V(w, "blocks.0.mlp.fc1.weight"), V(w, "blocks.0.mlp.fc1.bias")).Select(v => 0.5 * v * (1 + Erf(v / Math.Sqrt(2)))).ToArray();
                double[] down = Linear(up, V(w, "blocks.0.mlp.fc2.weight"), V(w, "blocks.0.mlp.fc2.bias"));

                // 4. the pooled token plus the learned patch position.
                int window = wy * side + wx;
                outputs[window] = first.Select((v, c) => v + down[c] + posPatch[c * side * side + window]).ToArray();
            }
        }

        return outputs;
    }

    private static float[] PointMap(Random rng, int h, int wd, int invalidCount)
    {
        float[] map = new float[3 * h * wd];
        for (int i = 0; i < map.Length; i++)
            map[i] = (float)((rng.NextDouble() - 0.5) * 4);
        for (int k = 0; k < invalidCount; k++)
            map[rng.Next(h * wd)] = k % 2 == 0 ? float.NaN : float.PositiveInfinity;
        return map;
    }

    [Fact]
    public void Embed_MatchesTheNaiveReferenceIncludingInvalidPointsAndANonSquareResize()
    {
        Random rng = new(21);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        SamPointPatchEmbed embed = new(Config, w);
        const int h = 6, wd = 10;
        float[] map = PointMap(rng, h, wd, 6);

        using Tensor tokens = embed.Embed(backend, map, h, wd);

        Assert.Equal(new TensorShape(4, 8), tokens.Shape);
        double[][] expected = Reference(w, map, h, wd);
        ReadOnlySpan<float> got = tokens.AsReadOnlySpan<float>();
        for (int i = 0; i < 4; i++)
        {
            for (int c = 0; c < 8; c++)
                Assert.True(Math.Abs(expected[i][c] - got[i * 8 + c]) < 2e-3 * (1 + Math.Abs(expected[i][c])), $"[{i},{c}] expected {expected[i][c]} got {got[i * 8 + c]}");
        }
    }

    [Fact]
    public void Embed_AnInvalidPointIsReplacedWholeSoItsOtherCoordinatesDoNotMatter()
    {
        Random rng = new(22);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        SamPointPatchEmbed embed = new(Config, w);
        float[] map = PointMap(rng, 8, 8, 0);
        map[0] = float.NaN;
        float[] changed = (float[])map.Clone();
        changed[64] = 3.5f;
        changed[128] = -2.25f;

        using Tensor a = embed.Embed(backend, map, 8, 8);
        using Tensor b = embed.Embed(backend, changed, 8, 8);

        Assert.Equal(a.AsReadOnlySpan<float>().ToArray(), b.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Embed_ExpDisparityStaysFiniteWhenInvalidPointsAreZeroedBeforeTheRemap()
    {
        // exp_disparity maps z = 0 to infinity or NaN, so an invalid point left at 0 would poison its window if it were not replaced by the token afterwards.
        SamPointPatchConfig config = Config with { Remap = SamPointRemapType.ExpDisparity };
        Random rng = new(25);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        SamPointPatchEmbed embed = new(config, w);
        float[] map = new float[3 * 8 * 8];
        for (int i = 0; i < 64; i++)
        {
            map[i] = (float)((rng.NextDouble() - 0.5) * 2);
            map[64 + i] = (float)((rng.NextDouble() - 0.5) * 2);
            map[128 + i] = (float)(0.5 + rng.NextDouble() * 2.5);
        }

        map[3] = float.NaN;
        map[64 + 9] = float.PositiveInfinity;
        map[128 + 40] = float.NegativeInfinity;

        using Tensor tokens = embed.Embed(backend, map, 8, 8);

        foreach (float value in tokens.AsReadOnlySpan<float>())
            Assert.True(float.IsFinite(value));
    }

    [Fact]
    public void Embed_RejectsAPointmapOfTheWrongLength()
    {
        Random rng = new(23);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        SamPointPatchEmbed embed = new(Config, w);

        Assert.Throws<ArgumentException>(() => embed.Embed(backend, new float[3 * 8 * 8 - 1], 8, 8));
    }

    [Fact]
    public void Constructor_RejectsMissingMisshapenWeightsAndABadConfig()
    {
        Random rng = new(24);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);

        Tensor kept = w["pos_embed"];
        w["pos_embed"] = Filled(rng, 1.0, 1, 8, 3, 3);
        Assert.Throws<InvalidOperationException>(() => new SamPointPatchEmbed(Config, w));
        w["pos_embed"] = kept;
        w.Remove("blocks.0.mlp.fc2.bias");
        Assert.Throws<InvalidOperationException>(() => new SamPointPatchEmbed(Config, w));

        Assert.Throws<ArgumentException>(() => new SamPointPatchEmbed(Config with { PatchSize = 3 }, w));
        Assert.Throws<ArgumentException>(() => new SamPointPatchEmbed(Config with { Heads = 3 }, w));
        SamPointPatchConfig.Released.Validate();
        Assert.Equal(1024, SamPointPatchConfig.Released.Windows);
        Assert.Equal(65, SamPointPatchConfig.Released.WindowTokens);
        Assert.Equal(1024, SamPointPatchConfig.Released.MlpHidden);
    }
}
