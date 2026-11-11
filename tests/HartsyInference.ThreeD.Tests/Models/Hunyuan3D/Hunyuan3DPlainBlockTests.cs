using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

/// <summary>The 2.1 mixture of experts against a host reference on tiny random weights: gate softmax, top-2 without renormalising, exact-erf GELU experts, and an ungated shared expert.</summary>
public sealed unsafe class Hunyuan3DPlainBlockTests
{
    private const int Width = 4, Heads = 2, Hidden = 8, Experts = 4, TopK = 2, Tokens = 6;

    [Fact]
    public void Moe_MatchesAHostReferenceOfSharedPlusTopTwoWeightedExperts()
    {
        Random random = new(7);
        Dictionary<string, Tensor> w = Weights(random);
        Hunyuan3DPlainBlock block = new(Width, Heads, skip: false, moe: true, Experts, TopK);
        block.LoadWeights(w, "b", halfWeights: false);
        float[] x = Values(random, Tokens * Width, 1f);

        using CpuBackend backend = new();
        Tensor input = Make(x, 1, Tokens, Width);
        using Tensor output = block.Moe(backend, input, Tokens);

        float[] expected = Reference(w, x);
        float[] actual = new float[expected.Length];
        new ReadOnlySpan<float>((float*)output.DataPointer, actual.Length).CopyTo(actual);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f, $"element {i}: expected {expected[i]}, got {actual[i]}");
    }

    [Fact]
    public void Regroup_ReadsEachHeadsQkvAsConsecutiveGroupsOfTheFusedRow()
    {
        // Upstream: cat(q, k, v) along the channel axis, then view (heads, 3 * headDim). Head h's "q" is fused channels
        // [h * 3d, h * 3d + d), its "k" the next d, its "v" the last d. Rows 0..W-1 are q, W..2W-1 k, 2W..3W-1 v.
        const int cols = 3, headDim = Width / Heads;
        float[][] source = [[.. Enumerable.Range(0, Width * cols).Select(i => 1f + i)], [.. Enumerable.Range(0, Width * cols).Select(i => 1000f + i)],
                            [.. Enumerable.Range(0, Width * cols).Select(i => 2000f + i)]];
        Tensor q = Make(source[0], Width, cols), k = Make(source[1], Width, cols), v = Make(source[2], Width, cols);
        Hunyuan3DPlainBlock block = new(Width, Heads, skip: false, moe: false, Experts, TopK);

        (Tensor rq, Tensor rk, Tensor? rv) = block.Regroup(q, k, v, 3);

        Tensor[] regrouped = [rq, rk, rv!];
        for (int part = 0; part < 3; part++)
        {
            for (int head = 0; head < Heads; head++)
            {
                for (int j = 0; j < headDim; j++)
                {
                    int fusedRow = head * 3 * headDim + part * headDim + j, sourceRow = fusedRow % Width;
                    float[] from = source[fusedRow / Width];
                    for (int c = 0; c < cols; c++)
                        Assert.Equal(from[sourceRow * cols + c], ((float*)regrouped[part].DataPointer)[(head * headDim + j) * cols + c]);
                }
            }
        }

        // Cross-attention: cat(k, v) viewed (heads, 2 * headDim).
        (Tensor ck, Tensor cv, _) = block.Regroup(q, k, null, 2);
        for (int head = 0; head < Heads; head++)
        {
            for (int j = 0; j < headDim; j++)
            {
                int kRow = head * 2 * headDim + j;
                Assert.Equal(source[kRow / Width][kRow % Width * cols], ((float*)ck.DataPointer)[(head * headDim + j) * cols]);
                int vRow = head * 2 * headDim + headDim + j;
                Assert.Equal(source[vRow / Width][vRow % Width * cols], ((float*)cv.DataPointer)[(head * headDim + j) * cols]);
            }
        }
    }

    // shared(x) + sum over the two highest gate scores of score * expert(x), the scores being the plain softmax values.
    private static float[] Reference(Dictionary<string, Tensor> w, float[] x)
    {
        float[] result = new float[Tokens * Width];
        for (int t = 0; t < Tokens; t++)
        {
            float[] token = x[(t * Width)..((t + 1) * Width)];
            double[] logits = new double[Experts];
            for (int e = 0; e < Experts; e++)
                logits[e] = Dot(w["b.moe.gate.weight"], e, token);
            double max = logits.Max(), sum = logits.Sum(l => Math.Exp(l - max));
            double[] scores = logits.Select(l => Math.Exp(l - max) / sum).ToArray();
            int[] top = Enumerable.Range(0, Experts).OrderByDescending(e => scores[e]).Take(TopK).ToArray();

            double[] y = Expert(w, "b.moe.shared_experts", token);
            foreach (int e in top)
            {
                double[] routed = Expert(w, $"b.moe.experts.{e}", token);
                for (int c = 0; c < Width; c++) y[c] += scores[e] * routed[c];
            }

            for (int c = 0; c < Width; c++) result[t * Width + c] = (float)y[c];
        }

        return result;
    }

    private static double[] Expert(Dictionary<string, Tensor> w, string prefix, float[] token)
    {
        double[] h = new double[Hidden];
        for (int j = 0; j < Hidden; j++)
        {
            double a = Dot(w[$"{prefix}.net.0.proj.weight"], j, token) + ((float*)w[$"{prefix}.net.0.proj.bias"].DataPointer)[j];
            h[j] = 0.5 * a * (1 + Erf(a / Math.Sqrt(2)));
        }

        double[] o = new double[Width];
        for (int c = 0; c < Width; c++)
        {
            float* row = (float*)w[$"{prefix}.net.2.weight"].DataPointer + c * Hidden;
            double acc = ((float*)w[$"{prefix}.net.2.bias"].DataPointer)[c];
            for (int j = 0; j < Hidden; j++) acc += row[j] * h[j];
            o[c] = acc;
        }

        return o;
    }

    private static double Dot(Tensor matrix, int row, float[] vector)
    {
        float* p = (float*)matrix.DataPointer + row * vector.Length;
        double acc = 0;
        for (int i = 0; i < vector.Length; i++) acc += p[i] * vector[i];
        return acc;
    }

    // Abramowitz-Stegun 7.1.26 is too coarse for a 1e-4 check; use the series/continued form via a high-precision erf.
    private static double Erf(double x)
    {
        double sign = Math.Sign(x), a = Math.Abs(x);
        if (a < 3)
        {
            double sum = a, term = a;
            for (int n = 1; n < 80; n++)
            {
                term *= -a * a / n;
                sum += term / (2 * n + 1);
            }

            return sign * 2 / Math.Sqrt(Math.PI) * sum;
        }

        return sign * (1 - Math.Exp(-a * a) / (a * Math.Sqrt(Math.PI)) * (1 - 1 / (2 * a * a)));
    }

    private static Dictionary<string, Tensor> Weights(Random random)
    {
        int headDim = Width / Heads;
        Dictionary<string, Tensor> w = [];
        void Add(string key, float scale, params long[] shape) => w[key] = Make(Values(random, (int)shape.Aggregate(1L, (a, b) => a * b), scale), shape);
        foreach (string n in new[] { "norm1", "norm2", "norm3" })
        {
            Add($"b.{n}.weight", 0.1f, Width);
            Add($"b.{n}.bias", 0.1f, Width);
        }

        foreach (string a in new[] { "attn1", "attn2" })
        {
            foreach (string p in new[] { "to_q", "to_k", "to_v" })
                Add($"b.{a}.{p}.weight", 0.3f, Width, Width);
            Add($"b.{a}.q_norm.weight", 0.1f, headDim);
            Add($"b.{a}.k_norm.weight", 0.1f, headDim);
            Add($"b.{a}.out_proj.weight", 0.3f, Width, Width);
            Add($"b.{a}.out_proj.bias", 0.1f, Width);
        }

        Add("b.moe.gate.weight", 1.5f, Experts, Width);
        foreach (string e in new[] { "shared_experts" }.Concat(Enumerable.Range(0, Experts).Select(i => $"experts.{i}")))
        {
            Add($"b.moe.{e}.net.0.proj.weight", 0.5f, Hidden, Width);
            Add($"b.moe.{e}.net.0.proj.bias", 0.2f, Hidden);
            Add($"b.moe.{e}.net.2.weight", 0.5f, Width, Hidden);
            Add($"b.moe.{e}.net.2.bias", 0.2f, Width);
        }

        return w;
    }

    private static float[] Values(Random random, int count, float scale) =>
        Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() * 2 - 1) * scale).ToArray();

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((float*)tensor.DataPointer, values.Length));
        return tensor;
    }
}
