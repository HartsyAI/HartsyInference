using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using HartsyInference.Vision.Dinov2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamDinoEmbedderTests
{
    // A tiny DINOv2 with registers: width 16, two layers, two heads, a 2 x 2 patch grid of 14-pixel patches (28 px input), four registers.
    private static readonly Dinov2Preset Tiny = new()
    {
        Name = "tiny", HiddenSize = 16, NumLayers = 2, NumHeads = 2, IntermediateSize = 32, ImageSize = 28, PatchSize = 14, NumRegisterTokens = 4,
    };

    private const int D = 16, Heads = 2, Hidden = 32, Size = 28, Patch = 14, Grid = 2, Regs = 4;
    private static readonly double[] Mean = [0.485, 0.456, 0.406];
    private static readonly double[] Std = [0.229, 0.224, 0.225];

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

    private static Dictionary<string, Tensor> HubWeights(Random rng)
    {
        Dictionary<string, Tensor> w = new()
        {
            ["patch_embed.proj.weight"] = Filled(rng, 0.2, D, 3, Patch, Patch),
            ["patch_embed.proj.bias"] = Filled(rng, 0.3, D),
            ["cls_token"] = Filled(rng, 1.0, 1, 1, D),
            ["pos_embed"] = Filled(rng, 1.0, 1, 1 + Grid * Grid, D),
            ["register_tokens"] = Filled(rng, 1.0, 1, Regs, D),
            ["mask_token"] = Filled(rng, 1.0, 1, D),
            ["norm.weight"] = Filled(rng, 1.0, D),
            ["norm.bias"] = Filled(rng, 0.4, D),
        };
        for (int i = 0; i < Tiny.NumLayers; i++)
        {
            string b = $"blocks.{i}.";
            w[b + "norm1.weight"] = Filled(rng, 1.0, D);
            w[b + "norm1.bias"] = Filled(rng, 0.4, D);
            w[b + "norm2.weight"] = Filled(rng, 1.0, D);
            w[b + "norm2.bias"] = Filled(rng, 0.4, D);
            w[b + "attn.qkv.weight"] = Filled(rng, 0.8, 3 * D, D);
            w[b + "attn.qkv.bias"] = Filled(rng, 0.4, 3 * D);
            w[b + "attn.proj.weight"] = Filled(rng, 0.8, D, D);
            w[b + "attn.proj.bias"] = Filled(rng, 0.4, D);
            w[b + "ls1.gamma"] = Filled(rng, 1.0, D);
            w[b + "ls2.gamma"] = Filled(rng, 1.0, D);
            w[b + "mlp.fc1.weight"] = Filled(rng, 0.8, Hidden, D);
            w[b + "mlp.fc1.bias"] = Filled(rng, 0.4, Hidden);
            w[b + "mlp.fc2.weight"] = Filled(rng, 0.8, D, Hidden);
            w[b + "mlp.fc2.bias"] = Filled(rng, 0.4, D);
        }

        return w;
    }

    private static double[] V(Dictionary<string, Tensor> w, string key) => w[key].AsReadOnlySpan<float>().ToArray().Select(f => (double)f).ToArray();

    private static double Erf(double x)
    {
        double sum = x, term = x;
        for (int n = 1; n < 80; n++)
        {
            term *= -x * x / n;
            sum += term / (2 * n + 1);
        }

        return 2 / Math.Sqrt(Math.PI) * sum;
    }

    private static double[] LayerNorm(double[] x, double[]? weight, double[]? bias, double eps)
    {
        double mean = x.Average(), variance = x.Select(v => (v - mean) * (v - mean)).Average();
        double inverse = 1 / Math.Sqrt(variance + eps);
        return x.Select((v, i) => (v - mean) * inverse * (weight?[i] ?? 1) + (bias?[i] ?? 0)).ToArray();
    }

    private static double[] Linear(double[] x, double[] weight, double[] bias)
    {
        int outputs = bias.Length;
        double[] y = new double[outputs];
        for (int o = 0; o < outputs; o++)
        {
            double sum = bias[o];
            for (int i = 0; i < x.Length; i++)
                sum += weight[o * x.Length + i] * x[i];
            y[o] = sum;
        }

        return y;
    }

    /// <summary>The upstream forward in double precision on a hub-named state dict: patch embed, class token and position, registers after the position, layer-scaled blocks.</summary>
    private static (double[][] Final, double[][] Prenorm) Reference(Dictionary<string, Tensor> w, double[] rgb)
    {
        // pixels: normalised planar [3, Size, Size]
        double[] w0 = V(w, "patch_embed.proj.weight"), b0 = V(w, "patch_embed.proj.bias"), cls = V(w, "cls_token"), pos = V(w, "pos_embed"), reg = V(w, "register_tokens");
        List<double[]> x = [];
        double[] first = new double[D];
        for (int c = 0; c < D; c++) first[c] = cls[c] + pos[c];
        x.Add(first);
        for (int r = 0; r < Regs; r++) x.Add(reg.Skip(r * D).Take(D).ToArray());
        for (int gy = 0; gy < Grid; gy++)
        {
            for (int gx = 0; gx < Grid; gx++)
            {
                double[] token = new double[D];
                for (int c = 0; c < D; c++)
                {
                    double sum = b0[c];
                    for (int ci = 0; ci < 3; ci++)
                        for (int ky = 0; ky < Patch; ky++)
                            for (int kx = 0; kx < Patch; kx++)
                                sum += w0[((c * 3 + ci) * Patch + ky) * Patch + kx] * rgb[(ci * Size + gy * Patch + ky) * Size + gx * Patch + kx];
                    token[c] = sum + pos[(1 + gy * Grid + gx) * D + c];
                }

                x.Add(token);
            }
        }

        int n = x.Count, headDim = D / Heads;
        for (int layer = 0; layer < Tiny.NumLayers; layer++)
        {
            string b = $"blocks.{layer}.";
            double[] qkvW = V(w, b + "attn.qkv.weight"), qkvB = V(w, b + "attn.qkv.bias"), projW = V(w, b + "attn.proj.weight"), projB = V(w, b + "attn.proj.bias");
            double[][] qkv = x.Select(t => Linear(LayerNorm(t, V(w, b + "norm1.weight"), V(w, b + "norm1.bias"), 1e-6), qkvW, qkvB)).ToArray();
            double[] ls1 = V(w, b + "ls1.gamma"), ls2 = V(w, b + "ls2.gamma");
            double[][] attended = new double[n][];
            for (int i = 0; i < n; i++)
            {
                attended[i] = new double[D];
                for (int h = 0; h < Heads; h++)
                {
                    double[] scores = new double[n];
                    for (int j = 0; j < n; j++)
                    {
                        double dot = 0;
                        for (int c = 0; c < headDim; c++) dot += qkv[i][h * headDim + c] * qkv[j][D + h * headDim + c];
                        scores[j] = dot / Math.Sqrt(headDim);
                    }

                    double max = scores.Max(), total = scores.Sum(s => Math.Exp(s - max));
                    for (int j = 0; j < n; j++)
                    {
                        double p = Math.Exp(scores[j] - max) / total;
                        for (int c = 0; c < headDim; c++) attended[i][h * headDim + c] += p * qkv[j][2 * D + h * headDim + c];
                    }
                }
            }

            for (int i = 0; i < n; i++)
            {
                double[] o = Linear(attended[i], projW, projB);
                for (int c = 0; c < D; c++) x[i][c] += ls1[c] * o[c];
            }

            for (int i = 0; i < n; i++)
            {
                double[] up = Linear(LayerNorm(x[i], V(w, b + "norm2.weight"), V(w, b + "norm2.bias"), 1e-6), V(w, b + "mlp.fc1.weight"), V(w, b + "mlp.fc1.bias"));
                double[] act = up.Select(v => 0.5 * v * (1 + Erf(v / Math.Sqrt(2)))).ToArray();
                double[] down = Linear(act, V(w, b + "mlp.fc2.weight"), V(w, b + "mlp.fc2.bias"));
                for (int c = 0; c < D; c++) x[i][c] += ls2[c] * down[c];
            }
        }

        double[][] prenorm = x.Select(t => LayerNorm(t, null, null, 1e-5)).ToArray();
        double[] nw = V(w, "norm.weight"), nb = V(w, "norm.bias");
        double[][] normed = x.Select(t => LayerNorm(t, nw, nb, 1e-6)).ToArray();
        double[][] final = [normed[0], .. normed.Skip(1 + Regs)];
        return (final, prenorm);
    }

    private static float[] Pixels(Random rng, int channels)
    {
        float[] p = new float[channels * Size * Size];
        for (int i = 0; i < p.Length; i++) p[i] = (float)rng.NextDouble();
        return p;
    }

    private static double[] Normalised(float[] pixels, int channels)
    {
        double[] rgb = new double[3 * Size * Size];
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < Size * Size; i++)
                rgb[c * Size * Size + i] = (pixels[(channels == 1 ? 0 : c) * Size * Size + i] - Mean[c]) / Std[c];
        return rgb;
    }

    private static void AssertClose(double[][] expected, Tensor actual, string what)
    {
        Assert.Equal(new TensorShape(1, expected.Length, D), actual.Shape);
        ReadOnlySpan<float> got = actual.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            for (int c = 0; c < D; c++)
                Assert.True(Math.Abs(expected[i][c] - got[i * D + c]) < 2e-3 * (1 + Math.Abs(expected[i][c])), $"{what} [{i},{c}] expected {expected[i][c]} got {got[i * D + c]}");
    }

    [Fact]
    public void StructureAndSlatTokens_MatchTheNaiveReferenceForAnImage()
    {
        Random rng = new(31);
        Dictionary<string, Tensor> w = HubWeights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        using SamDinoEmbedder dino = new(Tiny, w, "");
        float[] pixels = Pixels(rng, 3);
        (double[][] final, double[][] prenorm) = Reference(w, Normalised(pixels, 3));

        using Tensor structure = dino.StructureTokens(backend, pixels);
        using Tensor slat = dino.SlatTokens(backend, pixels);

        // The structure tokens drop the registers (class token, then the 4 patches); the SLAT tokens keep all 1 + 4 + 4 rows.
        AssertClose(final, structure, "structure");
        AssertClose(prenorm, slat, "slat");
    }

    [Fact]
    public void MaskTokens_RepeatTheMaskOverThreeChannelsBeforeTheNormalisation()
    {
        Random rng = new(32);
        Dictionary<string, Tensor> w = HubWeights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        using SamDinoEmbedder dino = new(Tiny, w, "");
        float[] mask = Pixels(rng, 1);
        (double[][] final, double[][] prenorm) = Reference(w, Normalised(mask, 1));

        using Tensor structure = dino.StructureTokensOfMask(backend, mask);
        using Tensor slat = dino.SlatTokensOfMask(backend, mask);

        AssertClose(final, structure, "mask structure");
        AssertClose(prenorm, slat, "mask slat");
    }

    [Fact]
    public void Constructor_ReadsKeysUnderAPrefixAndRejectsAMissingOrMisshapenWeight()
    {
        Random rng = new(33);
        Dictionary<string, Tensor> w = HubWeights(rng);
        using Owned owned = new(w.Values);
        Dictionary<string, Tensor> prefixed = w.ToDictionary(kv => "tower." + kv.Key, kv => kv.Value);

        using (SamDinoEmbedder fine = new(Tiny, prefixed, "tower.")) { }

        Dictionary<string, Tensor> broken = new(w);
        broken.Remove("blocks.1.ls2.gamma");
        Assert.Throws<InvalidOperationException>(() => new SamDinoEmbedder(Tiny, broken, ""));
        using Tensor wrong = Filled(rng, 1.0, 1, 3, D);
        broken = new Dictionary<string, Tensor>(w) { ["pos_embed"] = wrong };
        Assert.Throws<InvalidOperationException>(() => new SamDinoEmbedder(Tiny, broken, ""));
    }

    [Fact]
    public void Tokens_RejectAnInputOfTheWrongLength()
    {
        Random rng = new(34);
        Dictionary<string, Tensor> w = HubWeights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        using SamDinoEmbedder dino = new(Tiny, w, "");

        Assert.Throws<ArgumentException>(() => dino.StructureTokens(backend, new float[Size * Size]));
        Assert.Throws<ArgumentException>(() => dino.SlatTokensOfMask(backend, new float[3 * Size * Size]));
    }
}
