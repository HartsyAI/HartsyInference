using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Tests.VibeVoice;

/// <summary>Deterministic random weights for a tiny VibeVoice acoustic tokenizer, in the engine's own key layout.</summary>
internal static unsafe class VibeVoiceSyntheticWeights
{
    /// <summary>A small config with distinct strides, so a swapped ratio or channel width changes the output.</summary>
    internal static VibeVoiceTokenizerConfig TinyConfig { get; } = new()
    {
        EncoderNFilters = 4,
        DecoderNFilters = 4,
        EncoderRatios = [3, 2, 2, 2, 2, 2],
        DecoderRatios = [3, 2, 2, 2, 2, 2],
        EncoderDepths = "1-1-1-1-1-1-2",
        VaeDim = 8,
    };

    /// <summary>Hop length of <see cref="TinyConfig"/>: the product of its ratios.</summary>
    internal const int TinyHop = 96;

    internal static Dictionary<string, Tensor> Create(VibeVoiceTokenizerConfig config, string prefix, int seed, bool includeEncoder = true)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        int[] depths = [.. config.EncoderDepths.Split('-').Select(int.Parse)];
        int n = depths.Length;
        int nf = config.EncoderNFilters;
        if (includeEncoder)
        {
            int[] ratios = [.. config.EncoderRatios.Reverse()];
            Conv(w, rng, $"{prefix}.encoder.downsample_layers.0.0.conv.conv", nf, config.Channels, 7);
            for (int i = 1; i < n; i++)
            {
                Conv(w, rng, $"{prefix}.encoder.downsample_layers.{i}.0.conv.conv", nf << i, nf << (i - 1), ratios[i - 1] * 2);
            }
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < depths[i]; j++)
                {
                    Block(w, rng, $"{prefix}.encoder.stages.{i}.{j}", nf << i);
                }
            }
            Conv(w, rng, $"{prefix}.encoder.head.conv.conv", config.VaeDim, nf << (n - 1), 7);
        }
        int dn = config.DecoderNFilters;
        int[] decDepths = [.. depths.Reverse()];
        Conv(w, rng, $"{prefix}.decoder.upsample_layers.0.0.conv.conv", dn << (n - 1), config.VaeDim, 7);
        for (int i = 0; i < n - 1; i++)
        {
            ConvTranspose(w, rng, $"{prefix}.decoder.upsample_layers.{i + 1}.0.convtr.convtr",
                dn << (n - 1 - i), dn << (n - 2 - i), config.DecoderRatios![i] * 2);
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < decDepths[i]; j++)
            {
                Block(w, rng, $"{prefix}.decoder.stages.{i}.{j}", dn << (n - 1 - i));
            }
        }
        Conv(w, rng, $"{prefix}.decoder.head.conv.conv", config.Channels, dn, 7);
        return w;
    }

    private static void Block(Dictionary<string, Tensor> w, Random rng, string p, int dim)
    {
        Vec(w, rng, $"{p}.norm.weight", dim, 1f, 0.2f);
        Vec(w, rng, $"{p}.ffn_norm.weight", dim, 1f, 0.2f);
        Vec(w, rng, $"{p}.gamma", dim, 0.5f, 0.1f);
        Vec(w, rng, $"{p}.ffn_gamma", dim, 0.5f, 0.1f);
        Mat(w, rng, $"{p}.ffn.linear1.weight", [4 * dim, dim], 1f / MathF.Sqrt(dim));
        Mat(w, rng, $"{p}.ffn.linear1.bias", [4 * dim], 0.05f);
        Mat(w, rng, $"{p}.ffn.linear2.weight", [dim, 4 * dim], 1f / MathF.Sqrt(4 * dim));
        Mat(w, rng, $"{p}.ffn.linear2.bias", [dim], 0.05f);
        Mat(w, rng, $"{p}.mixer.conv.conv.conv.weight", [dim, 1, 7], 0.3f);
        Mat(w, rng, $"{p}.mixer.conv.conv.conv.bias", [dim], 0.05f);
    }

    private static void Conv(Dictionary<string, Tensor> w, Random rng, string p, int outCh, int inCh, int kernel)
    {
        Mat(w, rng, $"{p}.weight", [outCh, inCh, kernel], 1f / MathF.Sqrt(inCh * kernel));
        Mat(w, rng, $"{p}.bias", [outCh], 0.05f);
    }

    private static void ConvTranspose(Dictionary<string, Tensor> w, Random rng, string p, int inCh, int outCh, int kernel)
    {
        Mat(w, rng, $"{p}.weight", [inCh, outCh, kernel], 1f / MathF.Sqrt(inCh * kernel / 2));
        Mat(w, rng, $"{p}.bias", [outCh], 0.05f);
    }

    private static void Vec(Dictionary<string, Tensor> w, Random rng, string key, int n, float center, float spread)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++)
        {
            p[i] = center + spread * (float)(rng.NextDouble() * 2 - 1);
        }
        w[key] = t;
    }

    private static void Mat(Dictionary<string, Tensor> w, Random rng, string key, int[] shape, float scale)
    {
        Tensor t = new(new TensorShape(shape.Select(static d => (long)d).ToArray()), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            p[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        }
        w[key] = t;
    }
}
