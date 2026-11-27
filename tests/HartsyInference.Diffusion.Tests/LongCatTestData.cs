using HartsyInference.Core.Configuration;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A tiny LongCat DiT (2 blocks, hidden 128, 4 heads of 32) with random weights under the reference key names.</summary>
internal static class LongCatTestData
{
    /// <summary>Keeps a checkpoint's DiT in F32 for the scope: the tiny parity references are F32 runs, so their comparison pins the F32 load
    /// (numerics.longcatDitBf16 is on by default, the reference's BF16 load). A knob profile pushed on this thread's async flow, so parallel
    /// tests keep their own load dtype; the process-wide knob store is never written.</summary>
    public static IDisposable PinF32Dit() =>
        KnobProfile.Create("longcat-f32-dit").With(EngineKnobs.LongCatDitBf16, false).Push();

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
        DType? storage = null, string indexFileName = LongCatCheckpoint.IndexFileName)
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
                part[key] = storage is null || weights[key].DType == DType.I8 ? weights[key] : weights[key].CastTo(storage.Value);
                map[key] = shardNames[shard];
                total += part[key].DType.ComputeByteCount(part[key].ElementCount);
            }
            SafeTensorsWriter.Save(Path.Combine(directory, shardNames[shard]), part);
        }
        File.WriteAllText(Path.Combine(directory, indexFileName),
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

    /// <summary>The release's per-row INT8 layout of <paramref name="weights"/>: every rank-2 weight except <c>final_layer.linear</c> becomes
    /// <c>.weight_int8</c> (round(w / scale)) plus <c>.weight_scale</c> (max|row| / 127); everything else is shared as is.</summary>
    public static Dictionary<string, Tensor> QuantizePerRow(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new(StringComparer.Ordinal);
        foreach ((string key, Tensor weight) in weights)
        {
            if (weight.Shape.Rank != 2 || !key.EndsWith(".weight", StringComparison.Ordinal)
                || key.StartsWith("final_layer.linear", StringComparison.Ordinal)
                || key.EndsWith("norm.weight", StringComparison.Ordinal) || key.EndsWith("pre_crs_attn_norm.weight", StringComparison.Ordinal))
            {
                result[key] = weight;
                continue;
            }
            long rows = weight.Shape[0], cols = weight.Shape[1];
            Tensor quantized = new(weight.Shape, DType.I8);
            Tensor scales = new(new TensorShape(rows), DType.F32);
            ReadOnlySpan<float> w = weight.AsReadOnlySpan<float>();
            Span<sbyte> q = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, sbyte>(quantized.AsSpan<byte>());
            Span<float> s = scales.AsSpan<float>();
            for (long r = 0; r < rows; r++)
            {
                float max = 0f;
                for (long c = 0; c < cols; c++)
                    max = MathF.Max(max, MathF.Abs(w[(int)(r * cols + c)]));
                s[(int)r] = max / 127f;
                for (long c = 0; c < cols; c++)
                    q[(int)(r * cols + c)] = (sbyte)MathF.Round(w[(int)(r * cols + c)] / s[(int)r]);
            }
            string baseKey = key[..^".weight".Length];
            result[baseKey + ".weight_int8"] = quantized;
            result[baseKey + ".weight_scale"] = scales;
        }
        return result;
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

    /// <summary>The tiny avatar config: the base tiny shape plus the audio projection and the per-block audio step.</summary>
    public static LongCatConfig TinyAvatar { get; } = new()
    {
        HiddenSize = 128, Depth = 2, NumHeads = 4, InChannels = 4, OutChannels = 4, CaptionChannels = 48, AdalnTembedDim = 64,
        FrequencyEmbeddingSize = 64, PatchSize = (1, 2, 2), TextTokensZeroPad = true, AudioOutputDim = 16, AudioWindow = 5, AudioBlock = 3,
        AudioChannel = 6, AudioIntermediateDim = 16, AudioContextTokens = 4, AudioVaeScale = 4,
    };

    /// <summary>The config's base weights plus the avatar's audio projection and each block's audio step, with seeded values.</summary>
    public static Dictionary<string, Tensor> AvatarWeights(LongCatConfig config, int seed)
    {
        Dictionary<string, Tensor> weights = Weights(config, seed);
        Random rng = new(seed + 1);
        int h = config.HiddenSize, ctx = config.AudioContextTokens, outDim = config.AudioOutputDim, inter = config.AudioIntermediateDim;
        int seqLen = config.AudioWindow, seqLenVf = config.AudioWindow + config.AudioVaeScale - 1, blocks = config.AudioBlock, channels = config.AudioChannel;
        void Add(string key, params int[] dims) => weights[key] = SmallRandomTensor(rng, dims);
        Add("audio_proj.proj1.weight", inter, seqLen * blocks * channels);
        Add("audio_proj.proj1.bias", inter);
        Add("audio_proj.proj1_vf.weight", inter, seqLenVf * blocks * channels);
        Add("audio_proj.proj1_vf.bias", inter);
        Add("audio_proj.proj2.weight", inter, inter);
        Add("audio_proj.proj2.bias", inter);
        Add("audio_proj.proj3.weight", ctx * outDim, inter);
        Add("audio_proj.proj3.bias", ctx * outDim);
        Add("audio_proj.norm.weight", outDim);
        Add("audio_proj.norm.bias", outDim);
        for (int b = 0; b < config.Depth; b++)
        {
            string p = $"blocks.{b}.";
            Add(p + "audio_adaLN_modulation.1.weight", 3 * h, config.AdalnTembedDim);
            Add(p + "audio_adaLN_modulation.1.bias", 3 * h);
            Add(p + "pre_video_crs_attn_norm.weight", h);
            Add(p + "pre_video_crs_attn_norm.bias", h);
            Add(p + "audio_cross_attn.q_linear.weight", h, h);
            Add(p + "audio_cross_attn.q_linear.bias", h);
            Add(p + "audio_cross_attn.kv_linear.weight", 2 * h, outDim);
            Add(p + "audio_cross_attn.kv_linear.bias", 2 * h);
            Add(p + "audio_cross_attn.q_norm.weight", config.HeadDim);
            Add(p + "audio_cross_attn.k_norm.weight", config.HeadDim);
            Add(p + "audio_cross_attn.proj.weight", h, h);
            Add(p + "audio_cross_attn.proj.bias", h);
        }
        return weights;
    }

    /// <summary>Uniform values in [-1, 1) from a seeded generator.</summary>
    public static float[] RandomValues(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static Tensor SmallRandomTensor(Random rng, params int[] dims)
    {
        Tensor tensor = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return tensor;
    }

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
