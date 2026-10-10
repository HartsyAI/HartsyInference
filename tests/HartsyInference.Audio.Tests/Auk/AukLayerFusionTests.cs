using HartsyInference.Audio.Models.Auk;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Layer fusion vs a naive double-precision reference, plus layer-index sensitivity.</summary>
public sealed class AukLayerFusionTests
{
    private const int Hidden = 16;
    private const int Tokens = 3;
    private const int Layers = 5;

    internal static float[][] RandomLayers(int seed)
    {
        Random rng = new(seed);
        float[][] layers = new float[Layers][];
        for (int l = 0; l < Layers; l++)
        {
            layers[l] = new float[Tokens * Hidden];
            for (int i = 0; i < layers[l].Length; i++) layers[l][i] = (float)(rng.NextDouble() * 6 - 3 + l);
        }
        return layers;
    }

    internal static Tensor ToTensor(float[] data)
    {
        Tensor t = new(new TensorShape(1, Tokens, Hidden), DType.F32);
        data.CopyTo(t.AsSpan<float>());
        return t;
    }

    internal static double[] Reference(float[][] layers, float[] weights, float scale)
    {
        double max = weights.Max();
        double[] e = weights.Select(w => Math.Exp(w - max)).ToArray();
        double sum = e.Sum();
        double[] result = new double[Tokens * Hidden];
        for (int l = 0; l < layers.Length; l++)
        {
            for (int r = 0; r < Tokens; r++)
            {
                double mean = 0;
                for (int d = 0; d < Hidden; d++) mean += layers[l][r * Hidden + d];
                mean /= Hidden;
                double var = 0;
                for (int d = 0; d < Hidden; d++) var += Math.Pow(layers[l][r * Hidden + d] - mean, 2);
                var /= Hidden;
                double inv = 1.0 / Math.Sqrt(var + 1e-5);
                for (int d = 0; d < Hidden; d++)
                    result[r * Hidden + d] += e[l] / sum * (layers[l][r * Hidden + d] - mean) * inv;
            }
        }
        for (int i = 0; i < result.Length; i++) result[i] *= scale;
        return result;
    }

    private static float[] Run(CpuBackend backend, float[][] layers, float[] weights, float scale, bool viaTapForAllButLast)
    {
        using AukLayerFusion fusion = new(backend, weights, scale, Hidden);
        fusion.Begin(Tokens);
        for (int l = 0; l < Layers - 1; l++)
        {
            using Tensor h = ToTensor(layers[l]);
            if (viaTapForAllButLast) fusion.OnLayer(l, h); else fusion.Accumulate(l, h);
        }
        using Tensor rawLast = ToTensor(layers[^1]);
        if (viaTapForAllButLast) fusion.OnLayer(Layers - 1, rawLast);
        using Tensor result = fusion.Complete(rawLast);
        return result.AsSpan<float>().ToArray();
    }

    [Theory]
    [InlineData(false)]
    public void MatchesDoubleReference(bool viaTap)
    {
        float[][] layers = RandomLayers(7);
        float[] weights = [0.3f, -1.2f, 2.0f, 0.1f, -0.4f];
        float[] got = Run(new CpuBackend(), layers, weights, 0.37f, viaTap);
        double[] want = Reference(layers, weights, 0.37f);
        for (int i = 0; i < want.Length; i++) Assert.True(Math.Abs(want[i] - got[i]) < 2e-5, $"i={i} want {want[i]} got {got[i]}");
    }

    [Fact]
    public void FromCheckpoint_ReadsTopLevelKeys_AndRejectsOutOfOrderLayers()
    {
        using Tensor lw = new(new TensorShape(3), DType.F32);
        using Tensor ls = new(new TensorShape(1), DType.F32);
        lw.AsSpan<float>().Clear();
        ls.AsSpan<float>()[0] = 3f;
        Dictionary<string, Tensor> dict = new() { ["layer_weights"] = lw, ["layer_scale"] = ls };
        using AukLayerFusion fusion = AukLayerFusion.FromCheckpoint(new CpuBackend(), dict, Hidden);
        Assert.Equal(3, fusion.Layers);
        Assert.Equal(1.0, fusion.Coefficients[1], 6);
        fusion.Begin(Tokens);
        using Tensor h = ToTensor(new float[Tokens * Hidden]);
        Assert.Throws<ArgumentException>(() => fusion.Accumulate(1, h));
        Assert.Throws<KeyNotFoundException>(() => AukLayerFusion.FromCheckpoint(new CpuBackend(), new Dictionary<string, Tensor>(), Hidden));
    }
}
