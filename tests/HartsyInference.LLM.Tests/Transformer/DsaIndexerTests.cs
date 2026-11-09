using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Sparse;

namespace HartsyInference.LLM.Tests.Transformer;

public sealed class DsaIndexerTests
{
    private const int Heads = 4, HeadDim = 16, RopeDim = 8, TopK = 5, Dim = 12, QLora = 10, Window = 4, Keys = 30;

    [Fact]
    public void Select_MatchesScalarIndexerFormula()
    {
        Random rng = new(17);
        float[] wqB = Rand(rng, Heads * HeadDim * QLora), wk = Rand(rng, HeadDim * Dim), wp = Rand(rng, Heads * Dim);
        float[] gamma = Rand(rng, HeadDim, 0.5f, 1f), beta = Rand(rng, HeadDim, 0.2f);
        float[] x = Rand(rng, Keys * Dim), qLatent = Rand(rng, Keys * QLora);
        DeepSeekV41RopeTable rope = DeepSeekV41RopeTable.Build(RopeDim, Keys, 8e6, null);
        using Tensor wqBT = Make(wqB, Heads * HeadDim, QLora), wkT = Make(wk, HeadDim, Dim), wpT = Make(wp, Heads, Dim);
        using Tensor gammaT = Make(gamma, HeadDim), betaT = Make(beta, HeadDim);
        DsaIndexerSettings settings = new()
        {
            Heads = Heads, HeadDim = HeadDim, RopeDim = RopeDim, TopK = TopK,
            QueryEncoding = LatentEncoding.F32, KeyEncoding = LatentEncoding.F32,
        };
        DsaIndexerWeights weights = new() { WqB = wqBT, Wk = wkT, KeyNormWeight = gammaT, KeyNormBias = betaT, WeightsProj = wpT };
        using DsaIndexer indexer = new(new CpuBackend(), settings, weights, rope, Dim, QLora);
        using Tensor keyStore = new(new TensorShape(Keys, HeadDim), DType.F32);

        // keys arrive in two chunks, as a prefill then a continuation
        indexer.AppendKeys(x.AsSpan(0, 18 * Dim), 18, 0, keyStore);
        indexer.AppendKeys(x.AsSpan(18 * Dim), Keys - 18, 18, keyStore);
        int[] positions = [0, 3, 4, 9, 17, 22, 29];
        int[] limits = positions.Select(p => SparseWindowMerge.IndexerLimit(p, Window)).ToArray();
        float[] xq = positions.SelectMany(p => x.AsSpan(p * Dim, Dim).ToArray()).ToArray();
        float[] ql = positions.SelectMany(p => qLatent.AsSpan(p * QLora, QLora).ToArray()).ToArray();

        int[] picked = indexer.Select(xq, ql, positions, limits, keyStore, out int width);

        Assert.Equal(Math.Min(TopK, limits.Max()), width);
        float[][] k = Enumerable.Range(0, Keys).Select(s => Key(x.AsSpan(s * Dim, Dim), s, wk, gamma, beta, rope)).ToArray();
        for (int t = 0; t < positions.Length; t++)
        {
            int p = positions[t];
            float[] w = MatVec(wp, x.AsSpan(p * Dim, Dim), Heads, Dim);
            float[] scores = new float[limits[t]];
            for (int h = 0; h < Heads; h++)
            {
                float[] q = Rotated(MatVec(wqB.AsSpan(h * HeadDim * QLora, HeadDim * QLora), qLatent.AsSpan(p * QLora, QLora), HeadDim, QLora), p, rope);
                for (int s = 0; s < limits[t]; s++)
                    scores[s] += w[h] / MathF.Sqrt(HeadDim) / MathF.Sqrt(Heads) * MathF.Max(Dot(q, k[s]), 0f);
            }
            int[] expected = Enumerable.Repeat(-1, width).ToArray();
            int[] best = SparseTopK.Best(scores, Math.Min(TopK, limits[t])).Order().ToArray();
            best.CopyTo(expected, 0);
            Assert.Equal(expected, picked.AsSpan(t * width, width).ToArray());
        }
    }

    [Fact]
    public void Select_RefusesPositionsPastTheRopeTable()
    {
        Random rng = new(2);
        DeepSeekV41RopeTable rope = DeepSeekV41RopeTable.Build(RopeDim, 8, 8e6, null);
        using Tensor wqBT = Make(Rand(rng, Heads * HeadDim * QLora), Heads * HeadDim, QLora), wkT = Make(Rand(rng, HeadDim * Dim), HeadDim, Dim);
        using Tensor wpT = Make(Rand(rng, Heads * Dim), Heads, Dim), gammaT = Make(Rand(rng, HeadDim), HeadDim);
        DsaIndexerSettings settings = new()
        {
            Heads = Heads, HeadDim = HeadDim, RopeDim = RopeDim, TopK = TopK,
            QueryEncoding = LatentEncoding.F32, KeyEncoding = LatentEncoding.F32,
        };
        DsaIndexerWeights weights = new() { WqB = wqBT, Wk = wkT, KeyNormWeight = gammaT, WeightsProj = wpT };
        using DsaIndexer indexer = new(new CpuBackend(), settings, weights, rope, Dim, QLora);
        using Tensor keyStore = new(new TensorShape(16, HeadDim), DType.F32);

        Assert.Throws<ArgumentOutOfRangeException>(() => indexer.AppendKeys(Rand(rng, 4 * Dim), 4, 6, keyStore));
    }

    // LayerNorm(wk x) * gamma + beta, then NeoX rope on the leading RopeDim values, then the orthonormal Hadamard.
    private static float[] Key(ReadOnlySpan<float> x, int pos, float[] wk, float[] gamma, float[] beta, DeepSeekV41RopeTable rope)
    {
        float[] k = MatVec(wk, x, HeadDim, Dim);
        float mean = k.Average(), var = k.Select(v => (v - mean) * (v - mean)).Average();
        for (int i = 0; i < HeadDim; i++) k[i] = (k[i] - mean) / MathF.Sqrt(var + 1e-6f) * gamma[i] + beta[i];
        return Rotated(k, pos, rope);
    }

    private static float[] Rotated(float[] v, int pos, DeepSeekV41RopeTable rope)
    {
        int half = RopeDim / 2;
        float[] r = (float[])v.Clone();
        for (int i = 0; i < half; i++)
        {
            float c = rope.Cos[pos * half + i], s = rope.Sin[pos * half + i];
            r[i] = v[i] * c - v[i + half] * s;
            r[i + half] = v[i] * s + v[i + half] * c;
        }
        float[] y = new float[HeadDim];
        for (int j = 0; j < HeadDim; j++)
        {
            double sum = 0;
            for (int i = 0; i < HeadDim; i++) sum += r[i] * ((System.Numerics.BitOperations.PopCount((uint)(i & j)) & 1) == 0 ? 1 : -1);
            y[j] = (float)(sum / Math.Sqrt(HeadDim));
        }
        return y;
    }

    private static float[] MatVec(ReadOnlySpan<float> w, ReadOnlySpan<float> x, int rows, int cols)
    {
        float[] y = new float[rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) y[r] += w[r * cols + c] * x[c];
        return y;
    }

    private static float Dot(float[] a, float[] b)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static float[] Rand(Random rng, int n, float scale = 1f, float bias = 0f) =>
        Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale + bias)).ToArray();

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
