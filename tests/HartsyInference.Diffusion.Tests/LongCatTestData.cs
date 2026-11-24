using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A tiny LongCat DiT (2 blocks, hidden 128, 4 heads of 32) with random weights under the reference key names.</summary>
internal static class LongCatTestData
{
    public static LongCatConfig Tiny { get; } = new()
    {
        HiddenSize = 128, Depth = 2, NumHeads = 4, InChannels = 4, OutChannels = 4, CaptionChannels = 48, AdalnTembedDim = 64,
        FrequencyEmbeddingSize = 64, PatchSize = (1, 2, 2),
    };

    public static Dictionary<string, Tensor> Weights(LongCatConfig config, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach ((string key, TensorShape shape) in LongCatCheckpoint.ExpectedShapes(config))
        {
            bool normScale = key.EndsWith("norm.weight", StringComparison.Ordinal) || key.EndsWith("pre_crs_attn_norm.weight", StringComparison.Ordinal);
            Tensor t = new(shape, DType.F32);
            Span<float> span = t.AsSpan<float>();
            for (int i = 0; i < span.Length; i++)
                span[i] = normScale ? 1f + 0.2f * Gaussian(rng) : 0.07f * Gaussian(rng);
            weights[key] = t;
        }
        return weights;
    }

    public static Tensor Random(TensorShape shape, int seed, float scale = 1f)
    {
        Random rng = new(seed);
        Tensor t = new(shape, DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
            span[i] = scale * Gaussian(rng);
        return t;
    }

    public static Tensor Latent(LongCatConfig config, int frames, int height, int width, int seed) =>
        Random(new TensorShape([1L, config.InChannels, frames, height, width]), seed);

    public static Tensor Encoder(LongCatConfig config, int rows, int seed) => Random(new TensorShape(rows, config.CaptionChannels), seed);

    public static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    public static double RelRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        if (actual.Length != expected.Length)
            throw new ArgumentException($"Lengths differ: {actual.Length} vs {expected.Length}.");
        double diff = 0, reference = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double d = actual[i] - (double)expected[i];
            diff += d * d;
            reference += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(diff / Math.Max(reference, 1e-30));
    }

    private static float Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
