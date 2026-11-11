using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

/// <summary>Tiny-config checks of the point encoder against a plain double-precision transcription of the architecture. The reference shares this reading of it.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class UniRigShapeEncoderTests
{
    private static readonly UniRigShapeEncoderConfig Tiny = new() { Width = 16, Heads = 2, Layers = 2, NumFreqs = 2, PointFeats = 3, TokenNum = 4 };

    [Fact]
    public void Encode_MatchesAPlainReferenceAndIsDeterministicPerSeed()
    {
        Dictionary<string, Tensor> weights = UniRigTestWeights.Encoder(Tiny, "enc", seed: 5);
        try
        {
            (float[] points, float[] normals) = Cloud(64, seed: 9);
            using CpuBackend backend = new();
            using UniRigShapeEncoder encoder = new(backend, Tiny, weights, "enc");
            using Tensor result = encoder.Encode(points, normals, seed: 3);
            Assert.Equal(new TensorShape(1, Tiny.TokenNum, Tiny.Width), result.Shape);

            float[] expected = Reference(Tiny, weights, "enc", points, normals, seed: 3);
            ReadOnlySpan<float> actual = result.AsSpan<float>();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(float.IsFinite(actual[i]));
                Assert.True(Math.Abs(expected[i] - actual[i]) < 2e-3f * Math.Max(1f, Math.Abs(expected[i])), $"element {i}: {actual[i]} vs reference {expected[i]}");
            }

            using Tensor again = encoder.Encode(points, normals, seed: 3);
            Assert.Equal(actual.ToArray(), again.AsSpan<float>().ToArray());
            using Tensor other = encoder.Encode(points, normals, seed: 4);
            Assert.NotEqual(actual.ToArray(), other.AsSpan<float>().ToArray());
        }
        finally
        {
            UniRigTestWeights.Dispose(weights);
        }
    }

    [Fact]
    public void Constructor_RejectsAWeightWithTheWrongShape()
    {
        Dictionary<string, Tensor> weights = UniRigTestWeights.Encoder(Tiny, "enc", seed: 1);
        try
        {
            weights["enc.ln_post.weight"].Dispose();
            weights["enc.ln_post.weight"] = new Tensor(new TensorShape(Tiny.Width + 1), DType.F32);
            using CpuBackend backend = new();
            Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => new UniRigShapeEncoder(backend, Tiny, weights, "enc"));
        }
        finally
        {
            UniRigTestWeights.Dispose(weights);
        }
    }

    [Fact]
    public void Encode_RejectsMismatchedInputs()
    {
        Dictionary<string, Tensor> weights = UniRigTestWeights.Encoder(Tiny, "enc", seed: 1);
        try
        {
            using CpuBackend backend = new();
            using UniRigShapeEncoder encoder = new(backend, Tiny, weights, "enc");
            Assert.Throws<ArgumentException>(() => encoder.Encode(new float[9], new float[6], 0));
            Assert.Throws<ArgumentException>(() => encoder.Encode([], [], 0));
        }
        finally
        {
            UniRigTestWeights.Dispose(weights);
        }
    }

    private static (float[] Points, float[] Normals) Cloud(int count, int seed)
    {
        Random random = new(seed);
        float[] points = new float[3 * count];
        float[] normals = new float[3 * count];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = (float)(random.NextDouble() * 2 - 1);
            normals[i] = (float)(random.NextDouble() * 2 - 1);
        }

        return (points, normals);
    }

    private static float[] Reference(UniRigShapeEncoderConfig c, Dictionary<string, Tensor> w, string p, float[] points, float[] normals, int seed)
    {
        int n = points.Length / 3;
        int[] pool = UniRigPointSelection.Choose(n, 4 * c.TokenNum, seed);
        float[] poolPoints = new float[3 * pool.Length];
        float[] poolFeats = new float[3 * pool.Length];
        for (int i = 0; i < pool.Length; i++)
        {
            Array.Copy(points, 3 * pool[i], poolPoints, 3 * i, 3);
            Array.Copy(normals, 3 * pool[i], poolFeats, 3 * i, 3);
        }

        int[] picked = SurfaceSampler.FarthestPointIndices(poolPoints, c.TokenNum);
        double[][] data = Enumerable.Range(0, n).Select(i => InputProj(c, w, p, points, normals, i)).ToArray();
        double[][] x = picked.Select(i => InputProj(c, w, p, poolPoints, poolFeats, i)).ToArray();

        // cross attention block
        double[][] q = x.Select(r => LayerNorm(r, w, $"{p}.cross_attn.ln_1")).ToArray();
        double[][] kv = data.Select(r => LayerNorm(r, w, $"{p}.cross_attn.ln_2")).ToArray();
        double[][] qp = q.Select(r => Linear(r, w[$"{p}.cross_attn.attn.c_q.weight"], null)).ToArray();
        double[][] kvp = kv.Select(r => Linear(r, w[$"{p}.cross_attn.attn.c_kv.weight"], null)).ToArray();
        double[][] attn = Attention(c, qp, kvp.Select(r => Pick(r, c, 2, 0)).ToArray(), kvp.Select(r => Pick(r, c, 2, 1)).ToArray());
        x = x.Select((r, i) => Add(r, Linear(attn[i], w[$"{p}.cross_attn.attn.c_proj.weight"], w[$"{p}.cross_attn.attn.c_proj.bias"]))).ToArray();
        x = x.Select(r => Add(r, Mlp(LayerNorm(r, w, $"{p}.cross_attn.ln_3"), w, $"{p}.cross_attn.mlp"))).ToArray();

        for (int layer = 0; layer < c.Layers; layer++)
        {
            string b = $"{p}.self_attn.resblocks.{layer}";
            double[][] normed = x.Select(r => LayerNorm(r, w, $"{b}.ln_1")).ToArray();
            double[][] qkv = normed.Select(r => Linear(r, w[$"{b}.attn.c_qkv.weight"], null)).ToArray();
            double[][] a = Attention(c, qkv.Select(r => Pick(r, c, 3, 0)).ToArray(), qkv.Select(r => Pick(r, c, 3, 1)).ToArray(), qkv.Select(r => Pick(r, c, 3, 2)).ToArray());
            x = x.Select((r, i) => Add(r, Linear(a[i], w[$"{b}.attn.c_proj.weight"], w[$"{b}.attn.c_proj.bias"]))).ToArray();
            x = x.Select(r => Add(r, Mlp(LayerNorm(r, w, $"{b}.ln_2"), w, $"{b}.mlp"))).ToArray();
        }

        return x.Select(r => LayerNorm(r, w, $"{p}.ln_post")).SelectMany(r => r.Select(v => (float)v)).ToArray();
    }

    private static double[] InputProj(UniRigShapeEncoderConfig c, Dictionary<string, Tensor> w, string p, float[] points, float[] feats, int i)
    {
        List<double> row = [points[3 * i], points[3 * i + 1], points[3 * i + 2]];
        List<double> s = [];
        List<double> co = [];
        for (int axis = 0; axis < 3; axis++)
        {
            for (int band = 0; band < c.NumFreqs; band++)
            {
                double angle = points[3 * i + axis] * Math.Pow(2, band);
                s.Add(Math.Sin(angle));
                co.Add(Math.Cos(angle));
            }
        }

        row.AddRange(s);
        row.AddRange(co);
        row.AddRange([feats[3 * i], feats[3 * i + 1], feats[3 * i + 2]]);
        return Linear([.. row], w[$"{p}.input_proj.weight"], w[$"{p}.input_proj.bias"]);
    }

    private static double[] Linear(double[] input, Tensor weight, Tensor? bias)
    {
        int outDim = (int)weight.Shape[0];
        int inDim = (int)weight.Shape[1];
        ReadOnlySpan<float> wd = weight.AsSpan<float>();
        double[] output = new double[outDim];
        for (int o = 0; o < outDim; o++)
        {
            double sum = bias is null ? 0 : bias.AsSpan<float>()[o];
            for (int k = 0; k < inDim; k++)
            {
                sum += wd[o * inDim + k] * input[k];
            }

            output[o] = sum;
        }

        return output;
    }

    private static double[] LayerNorm(double[] input, Dictionary<string, Tensor> w, string path)
    {
        float[] g = w[$"{path}.weight"].AsSpan<float>().ToArray();
        float[] b = w[$"{path}.bias"].AsSpan<float>().ToArray();
        double mean = input.Average();
        double variance = input.Select(v => (v - mean) * (v - mean)).Average();
        double scale = 1.0 / Math.Sqrt(variance + 1e-5);
        return input.Select((v, i) => (v - mean) * scale * g[i] + b[i]).ToArray();
    }

    private static double[] Mlp(double[] input, Dictionary<string, Tensor> w, string path)
    {
        double[] hidden = Linear(input, w[$"{path}.c_fc.weight"], w[$"{path}.c_fc.bias"]).Select(Gelu).ToArray();
        return Linear(hidden, w[$"{path}.c_proj.weight"], w[$"{path}.c_proj.bias"]);
    }

    private static double Gelu(double x) => 0.5 * x * (1 + Erf(x / Math.Sqrt(2)));

    private static double Erf(double x)
    {
        // Abramowitz & Stegun 7.1.26, max error 1.5e-7.
        double sign = Math.Sign(x);
        x = Math.Abs(x);
        double t = 1 / (1 + 0.3275911 * x);
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }

    private static double[] Add(double[] a, double[] b) => a.Select((v, i) => v + b[i]).ToArray();

    /// <summary>Slice t of k head-interleaved chunks: head h occupies [h*k*d, (h+1)*k*d), and chunk t sits at offset t*d inside it.</summary>
    private static double[] Pick(double[] row, UniRigShapeEncoderConfig c, int k, int t)
    {
        int d = c.Width / c.Heads;
        double[] result = new double[c.Width];
        for (int h = 0; h < c.Heads; h++)
        {
            for (int j = 0; j < d; j++)
            {
                result[h * d + j] = row[h * k * d + t * d + j];
            }
        }

        return result;
    }

    private static double[][] Attention(UniRigShapeEncoderConfig c, double[][] q, double[][] k, double[][] v)
    {
        int d = c.Width / c.Heads;
        double[][] output = q.Select(_ => new double[c.Width]).ToArray();
        for (int h = 0; h < c.Heads; h++)
        {
            for (int i = 0; i < q.Length; i++)
            {
                double[] scores = new double[k.Length];
                for (int j = 0; j < k.Length; j++)
                {
                    double dot = 0;
                    for (int e = 0; e < d; e++)
                    {
                        dot += q[i][h * d + e] * k[j][h * d + e];
                    }

                    scores[j] = dot / Math.Sqrt(d);
                }

                double max = scores.Max();
                double[] weights = scores.Select(s => Math.Exp(s - max)).ToArray();
                double total = weights.Sum();
                for (int j = 0; j < k.Length; j++)
                {
                    for (int e = 0; e < d; e++)
                    {
                        output[i][h * d + e] += weights[j] / total * v[j][h * d + e];
                    }
                }
            }
        }

        return output;
    }
}
