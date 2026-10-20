using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>T5's attention width is heads × d_kv, which equals d_model in every shipped size except T5-11B (128 heads of
/// 128 over d_model 1024). The block sized its Q/K/V buffers by d_model.</summary>
public sealed class T5BlockInnerWidthTests
{
    private const int DModel = 8, Heads = 2, HeadDim = 6, Seq = 5, Inner = Heads * HeadDim;

    [Fact]
    public void AttentionWiderThanTheModel_MatchesTheReference()
    {
        Random rng = new(11);
        float[] R(int n) { float[] a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
        float[] x = R(Seq * DModel), norm = R(DModel), wq = R(Inner * DModel), wk = R(Inner * DModel), wv = R(Inner * DModel), wo = R(DModel * Inner);
        Tensor T(float[] v, params long[] dims) { Tensor t = new(new TensorShape(dims), DType.F32); v.CopyTo(t.AsSpan<float>()); return t; }
        Dictionary<string, Tensor> w = new()
        {
            ["b.layer.0.layer_norm.weight"] = T(norm, DModel),
            ["b.layer.0.SelfAttention.q.weight"] = T(wq, Inner, DModel),
            ["b.layer.0.SelfAttention.k.weight"] = T(wk, Inner, DModel),
            ["b.layer.0.SelfAttention.v.weight"] = T(wv, Inner, DModel),
            ["b.layer.0.SelfAttention.o.weight"] = T(wo, DModel, Inner),
            // A zero FFN leaves the block's output as input + attention, which the reference below computes.
            ["b.layer.1.layer_norm.weight"] = T(new float[DModel], DModel),
            ["b.layer.1.DenseReluDense.wi_0.weight"] = T(new float[16 * DModel], 16, DModel),
            ["b.layer.1.DenseReluDense.wi_1.weight"] = T(new float[16 * DModel], 16, DModel),
            ["b.layer.1.DenseReluDense.wo.weight"] = T(new float[DModel * 16], DModel, 16),
        };
        T5Block block = new(DModel, Heads, HeadDim, 1e-6f, attentionScale: 1f);
        block.LoadWeights(w, "b");
        using CpuBackend backend = new();
        using Tensor input = T(x, 1, Seq, DModel);
        using Tensor bias = new(new TensorShape(1, Heads, Seq, Seq), DType.F32);
        using Tensor output = block.Forward(backend, input, bias, null);

        double[] expected = Reference(x, norm, wq, wk, wv, wo);
        ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4, $"[{i}] {actual[i]} vs {expected[i]}");
        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static double[] Reference(float[] x, float[] norm, float[] wq, float[] wk, float[] wv, float[] wo)
    {
        double[] Proj(double[] a, float[] wt, int outF, int inF)
        {
            double[] r = new double[Seq * outF];
            for (int s = 0; s < Seq; s++)
                for (int o = 0; o < outF; o++)
                {
                    double acc = 0;
                    for (int i = 0; i < inF; i++) acc += a[s * inF + i] * wt[o * inF + i];
                    r[s * outF + o] = acc;
                }
            return r;
        }
        double[] normed = new double[Seq * DModel];
        for (int s = 0; s < Seq; s++)
        {
            double ms = 0;
            for (int c = 0; c < DModel; c++) ms += (double)x[s * DModel + c] * x[s * DModel + c];
            double inv = 1 / Math.Sqrt(ms / DModel + 1e-6);
            for (int c = 0; c < DModel; c++) normed[s * DModel + c] = x[s * DModel + c] * inv * norm[c];
        }
        double[] q = Proj(normed, wq, Inner, DModel), k = Proj(normed, wk, Inner, DModel), v = Proj(normed, wv, Inner, DModel);
        double[] merged = new double[Seq * Inner];
        for (int h = 0; h < Heads; h++)
            for (int i = 0; i < Seq; i++)
            {
                double[] p = new double[Seq];
                double max = double.NegativeInfinity, sum = 0;
                for (int j = 0; j < Seq; j++)
                {
                    double dot = 0;
                    for (int d = 0; d < HeadDim; d++) dot += q[i * Inner + h * HeadDim + d] * k[j * Inner + h * HeadDim + d];
                    p[j] = dot;
                    max = Math.Max(max, dot);
                }
                for (int j = 0; j < Seq; j++) { p[j] = Math.Exp(p[j] - max); sum += p[j]; }
                for (int d = 0; d < HeadDim; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < Seq; j++) acc += p[j] * v[j * Inner + h * HeadDim + d];
                    merged[i * Inner + h * HeadDim + d] = acc / sum;
                }
            }
        double[] o = Proj(merged, wo, DModel, Inner);
        double[] y = new double[Seq * DModel];
        for (int i = 0; i < y.Length; i++) y[i] = x[i] + o[i];
        return y;
    }
}
