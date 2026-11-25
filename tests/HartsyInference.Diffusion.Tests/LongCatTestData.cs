using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;

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
            bool normScale = key.EndsWith("norm.weight", StringComparison.Ordinal)
                || key.EndsWith("pre_crs_attn_norm.weight", StringComparison.Ordinal);
            Tensor t = new(shape, DType.F32);
            Span<float> span = t.AsSpan<float>();
            for (int i = 0; i < span.Length; i++)
                span[i] = normScale ? 1f + 0.2f * Gaussian(rng) : 0.07f * Gaussian(rng);
            weights[key] = t;
        }
        return weights;
    }

    /// <summary>Writes <paramref name="weights"/> as the two-shard diffusers-named index the real checkpoints use, optionally
    /// leaving keys out or storing another dtype.</summary>
    public static void WriteCheckpoint(string directory, Dictionary<string, Tensor> weights, bool writeConfig = true, Func<string, bool>? skip = null,
        DType? storage = null)
    {
        string[] keys = weights.Keys.Order(StringComparer.Ordinal).Where(key => skip?.Invoke(key) != true).ToArray();
        string[] shardNames = ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"];
        Dictionary<string, string> map = [];
        long total = 0;
        for (int shard = 0; shard < 2; shard++)
        {
            Dictionary<string, Tensor> part = [];
            foreach (string key in keys.Where((_, i) => i % 2 == shard))
            {
                part[key] = storage is null ? weights[key] : weights[key].CastTo(storage.Value);
                map[key] = shardNames[shard];
                total += part[key].DType.ComputeByteCount(part[key].ElementCount);
            }
            SafeTensorsWriter.Save(Path.Combine(directory, shardNames[shard]), part);
        }
        File.WriteAllText(Path.Combine(directory, LongCatCheckpoint.IndexFileName),
            JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["metadata"] = new Dictionary<string, long> { ["total_size"] = total }, ["weight_map"] = map,
            }));
        if (writeConfig)
        {
            File.WriteAllText(Path.Combine(directory, "config.json"), """
                {"in_channels":4,"out_channels":4,"hidden_size":128,"depth":2,"num_heads":4,"caption_channels":48,"mlp_ratio":4,
                 "adaln_tembed_dim":64,"frequency_embedding_size":64,"patch_size":[1,2,2],"text_tokens_zero_pad":true}
                """);
        }
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
