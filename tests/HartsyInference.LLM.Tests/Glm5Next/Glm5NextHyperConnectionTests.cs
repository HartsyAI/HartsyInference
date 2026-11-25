using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Transformer.Mhc;
using Xunit;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The <see cref="HyperConnectionConfig.GlmFlash"/> preset against a double-precision port of transformers'
/// <c>Glm5NextTextHyperConnection.forward</c> and the decoder layer's expand (<c>post * sub + comb^T @ residual</c>) at commit 52bac5d0.</summary>
public sealed class Glm5NextHyperConnectionTests
{
    private const int Hc = 4, Dim = 6, Tokens = 3, Iters = 20;
    private const float HcEps = 1e-6f, RmsEps = 1e-5f;

    [Fact]
    public void MixesCollapseAndExpand_MatchTheTransformersFormulas()
    {
        Random rng = new(11);
        int mix = (2 + Hc) * Hc, flat = Hc * Dim;
        float[] fn = Random32(rng, mix * flat, 0.4f), bias = Random32(rng, mix, 0.5f), scale = [0.7f, 1.3f, 0.9f];
        float[] x = Random32(rng, Tokens * flat, 1.5f), sub = Random32(rng, Tokens * Dim, 1f);
        HyperConnection hc = new(new CpuBackend(), HyperConnectionConfig.GlmFlash(Hc, Dim, Iters, HcEps, RmsEps), fn, scale, bias);

        float[] pre = new float[Tokens * Hc], post = new float[Tokens * Hc], comb = new float[Tokens * Hc * Hc];
        hc.Mixes(x, Tokens, pre, post, comb);
        float[] collapsed = new float[Tokens * Dim], expanded = new float[Tokens * flat];
        hc.Collapse(x, pre, Tokens, collapsed);
        hc.Expand(sub, x, post, comb, Tokens, expanded);

        for (int t = 0; t < Tokens; t++)
        {
            (double[] preR, double[] postR, double[,] combR) = Reference(x.AsSpan(t * flat, flat).ToArray(), fn, bias, scale);
            for (int i = 0; i < Hc; i++)
            {
                Assert.Equal(preR[i], pre[t * Hc + i], 5);
                Assert.Equal(postR[i], post[t * Hc + i], 5);
                for (int j = 0; j < Hc; j++) Assert.Equal(combR[i, j], comb[(t * Hc + i) * Hc + j], 5);
            }
            for (int d = 0; d < Dim; d++)
            {
                double c = 0;
                for (int i = 0; i < Hc; i++) c += preR[i] * x[(t * Hc + i) * Dim + d];
                Assert.Equal(c, collapsed[t * Dim + d], 4);
                for (int i = 0; i < Hc; i++)
                {
                    double e = postR[i] * sub[t * Dim + d];
                    for (int j = 0; j < Hc; j++) e += combR[j, i] * x[(t * Hc + j) * Dim + d];
                    Assert.Equal(e, expanded[(t * Hc + i) * Dim + d], 4);
                }
            }
        }
    }

    // Port of Glm5NextTextHyperConnection.forward: unweighted RMS over the flattened streams at rms_norm_eps, one projection, then
    // pre = sigmoid + hc_eps, post = 2 sigmoid, comb = softmax(rows) + eps, a column pass, and iters - 1 row-then-column passes.
    private static (double[] Pre, double[] Post, double[,] Comb) Reference(float[] x, float[] fn, float[] bias, float[] scale)
    {
        int flat = Hc * Dim, mix = (2 + Hc) * Hc;
        double ms = x.Sum(v => (double)v * v) / flat, inv = 1.0 / Math.Sqrt(ms + RmsEps);
        double[] m = new double[mix];
        for (int o = 0; o < mix; o++)
            for (int i = 0; i < flat; i++) m[o] += x[i] * inv * fn[o * flat + i];
        double[] pre = new double[Hc], post = new double[Hc];
        for (int i = 0; i < Hc; i++)
        {
            pre[i] = Sigmoid(m[i] * scale[0] + bias[i]) + HcEps;
            post[i] = 2 * Sigmoid(m[Hc + i] * scale[1] + bias[Hc + i]);
        }
        double[,] c = new double[Hc, Hc];
        for (int r = 0; r < Hc; r++)
        {
            double[] row = new double[Hc];
            for (int k = 0; k < Hc; k++) row[k] = m[2 * Hc + r * Hc + k] * scale[2] + bias[2 * Hc + r * Hc + k];
            double max = row.Max(), sum = row.Sum(v => Math.Exp(v - max));
            for (int k = 0; k < Hc; k++) c[r, k] = Math.Exp(row[k] - max) / sum + HcEps;
        }
        Columns(c);
        for (int it = 1; it < Iters; it++)
        {
            for (int r = 0; r < Hc; r++)
            {
                double s = 0;
                for (int k = 0; k < Hc; k++) s += c[r, k];
                for (int k = 0; k < Hc; k++) c[r, k] /= s + HcEps;
            }
            Columns(c);
        }
        return (pre, post, c);
    }

    private static void Columns(double[,] c)
    {
        for (int k = 0; k < Hc; k++)
        {
            double s = 0;
            for (int r = 0; r < Hc; r++) s += c[r, k];
            for (int r = 0; r < Hc; r++) c[r, k] /= s + HcEps;
        }
    }

    private static double Sigmoid(double v) => 1.0 / (1.0 + Math.Exp(-v));

    private static float[] Random32(Random rng, int n, float amp)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        return v;
    }
}
