using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converter for Lumina-Image-2.0 (Alpha-VLLM) safetensors checkpoints. The diffusers and ComfyUI single-file distributions both use the same key naming as the upstream <c>Lumina2Transformer2DModel</c>: separate <c>attn.to_q/to_k/to_v.weight</c>, <c>attn.to_out.0.weight</c>, <c>attn.norm_q/k.weight</c>, <c>norm1.norm.weight</c> + <c>norm1.linear.{weight,bias}</c> (or just <c>norm1.weight</c> on context_refiner), <c>norm2.weight</c>, <c>ffn_norm{1,2}.weight</c>, <c>feed_forward.linear_{1,2,3}.weight</c>, <c>x_embedder.{weight,bias}</c>, <c>time_caption_embed.{caption_embedder.{0,1},timestep_embedder.linear_{1,2}}.*</c>, and <c>norm_out.linear_{1,2}.{weight,bias}</c>. This converter is mostly passthrough — its job is to fold per-tensor weight scales into <see cref="Tensor.Fp8ScaleFactor"/> (when shipped with ComfyUI <c>fp8_scaled</c> metadata) and partition transformer/VAE/text-encoder buckets.</summary>
public sealed class Lumina2CheckpointConverter
{
    /// <summary>Result of partitioning a Lumina-Image-2.0 single-file safetensors checkpoint.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Lumina-Image-2.0 transformer weights — pass directly to <c>Lumina2Transformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>VAE weights (Flux VAE; usually empty when the transformer ships standalone).</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }

        /// <summary>Gemma 2 2B text encoder weights (usually empty — text encoder ships separately).</summary>
        public required Dictionary<string, Tensor> TextEncoder { get; init; }

        /// <summary>True if any transformer linear weight is FP8 — pipeline should preload via the FP8 path.</summary>
        public required bool IsFp8Mix { get; init; }
    }

    /// <summary>Loads and partitions a Lumina-Image-2.0 checkpoint: a single file, or one shard of a diffusers multi-shard release (detected via a sibling <c>*.safetensors.index.json</c> in the same directory — the real <c>Alpha-VLLM/Lumina-Image-2.0</c> diffusers weights ship as 2 shards; every other shard alongside <paramref name="checkpointPath"/> is merged in too).</summary>
    public static (ConvertedWeights weights, IReadOnlyList<SafeTensorsLoader> loaders) LoadAndConvert(string checkpointPath)
    {
        string? dir = Path.GetDirectoryName(checkpointPath);
        bool isMultiShard = !string.IsNullOrEmpty(dir) && Directory.GetFiles(dir, "*.safetensors.index.json").Length > 0;

        List<SafeTensorsLoader> loaders = new();
        Dictionary<string, Tensor> merged = new();
        try
        {
            string[] shardPaths = isMultiShard
                ? Directory.GetFiles(dir!, "*.safetensors").OrderBy(p => p, StringComparer.Ordinal).ToArray()
                : new[] { checkpointPath };
            foreach (string shardPath in shardPaths)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shardPath);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
                    merged[kv.Key] = kv.Value;
            }
        }
        catch
        {
            foreach (SafeTensorsLoader loader in loaders) loader.Dispose();
            throw;
        }

        ConvertedWeights converted = Convert(merged);
        return (converted, loaders);
    }

    /// <summary>Partitions a flat dict of Lumina-Image-2.0 safetensors keys.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> dequanted = ApplyFp8WeightScales(allWeights);

        Dictionary<string, Tensor> transformer = new(dequanted.Count);
        Dictionary<string, Tensor> vae = new();
        Dictionary<string, Tensor> textEncoder = new();

        foreach (KeyValuePair<string, Tensor> kvp in dequanted)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            string transformerKey = key;
            if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                transformerKey = key["model.diffusion_model.".Length..];
            else if (key.StartsWith("transformer.", StringComparison.Ordinal))
                transformerKey = key["transformer.".Length..];

            if (IsTransformerKey(transformerKey))
            {
                transformer[transformerKey] = tensor;
                continue;
            }

            if (key.StartsWith("vae.", StringComparison.Ordinal) ||
                key.StartsWith("first_stage_model.", StringComparison.Ordinal))
            {
                vae[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.", StringComparison.Ordinal))
            {
                textEncoder[key] = tensor;
            }
        }

        bool isFp8Mix = DetectFp8Mix(transformer);

        return new ConvertedWeights
        {
            Transformer = transformer,
            Vae = vae,
            TextEncoder = textEncoder,
            IsFp8Mix = isFp8Mix,
        };
    }

    /// <summary>Auto-detects layer counts and FP8 status from a transformer dict.</summary>
    public static (int numLayers, int numRefinerLayers, int hidden, int qOutDim, int kvOutDim, int ffnDim, bool isFp8Mix) DetectArchitecture(
        IReadOnlyDictionary<string, Tensor> transformerWeights)
    {
        int maxLayer = -1;
        int maxNoiseRefiner = -1;
        int maxContextRefiner = -1;

        foreach (string key in transformerWeights.Keys)
        {
            if (key.StartsWith("layers.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 7);
                if (dot > 0 && int.TryParse(key.AsSpan(7, dot - 7), out int idx) && idx > maxLayer)
                    maxLayer = idx;
            }
            else if (key.StartsWith("noise_refiner.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 14);
                if (dot > 0 && int.TryParse(key.AsSpan(14, dot - 14), out int idx) && idx > maxNoiseRefiner)
                    maxNoiseRefiner = idx;
            }
            else if (key.StartsWith("context_refiner.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 16);
                if (dot > 0 && int.TryParse(key.AsSpan(16, dot - 16), out int idx) && idx > maxContextRefiner)
                    maxContextRefiner = idx;
            }
        }

        int hidden = 2304;
        int qOutDim = 2304;
        int kvOutDim = 768;
        int ffnDim = 6144;

        if (transformerWeights.TryGetValue("layers.0.attn.to_q.weight", out Tensor? toQ))
        {
            hidden = (int)toQ.Shape[1];
            qOutDim = (int)toQ.Shape[0];
        }
        if (transformerWeights.TryGetValue("layers.0.attn.to_k.weight", out Tensor? toK))
            kvOutDim = (int)toK.Shape[0];
        if (transformerWeights.TryGetValue("layers.0.feed_forward.linear_1.weight", out Tensor? linear1))
            ffnDim = (int)linear1.Shape[0];

        int numRefiner = Math.Min(maxNoiseRefiner, maxContextRefiner) + 1;
        if (numRefiner <= 0) numRefiner = 2;

        return (maxLayer + 1, numRefiner, hidden, qOutDim, kvOutDim, ffnDim, DetectFp8Mix(transformerWeights));
    }

    private static bool IsTransformerKey(string key)
    {
        return key.StartsWith("layers.", StringComparison.Ordinal)
            || key.StartsWith("noise_refiner.", StringComparison.Ordinal)
            || key.StartsWith("context_refiner.", StringComparison.Ordinal)
            || key.StartsWith("time_caption_embed.", StringComparison.Ordinal)
            || key.StartsWith("x_embedder.", StringComparison.Ordinal)
            || key.StartsWith("norm_out.", StringComparison.Ordinal);
    }

    private static bool DetectFp8Mix(IReadOnlyDictionary<string, Tensor> transformer)
    {
        if (transformer.TryGetValue("layers.0.attn.to_q.weight", out Tensor? probe))
            return probe.DType == DType.F8E4M3 || probe.DType == DType.F8E5M2;

        foreach (Tensor t in transformer.Values)
        {
            if (t.DType == DType.F8E4M3 || t.DType == DType.F8E5M2)
                return true;
        }
        return false;
    }

    /// <summary>Folds ComfyUI <c>fp8_scaled</c> per-tensor scale companions into <see cref="Tensor.Fp8ScaleFactor"/>. Suffix used here is <c>.weight_scale</c> (matches ComfyUI's convention for Lumina 2.0 quants when distributed). Also drops <c>.comfy_quant</c> metadata blobs.</summary>
    private static unsafe Dictionary<string, Tensor> ApplyFp8WeightScales(Dictionary<string, Tensor> source)
    {
        Dictionary<string, Tensor> scales = new();
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            if (kvp.Key.EndsWith(".weight_scale", StringComparison.Ordinal))
            {
                string baseKey = kvp.Key[..^".weight_scale".Length];
                scales[baseKey] = kvp.Value;
            }
        }
        if (scales.Count == 0)
            return source;

        Dictionary<string, Tensor> result = new(source.Count - 2 * scales.Count);
        foreach (KeyValuePair<string, Tensor> kvp in source)
        {
            if (kvp.Key.EndsWith(".weight_scale", StringComparison.Ordinal) ||
                kvp.Key.EndsWith(".comfy_quant", StringComparison.Ordinal))
            {
                continue;
            }

            if (kvp.Value.DType == DType.F8E4M3 &&
                kvp.Key.EndsWith(".weight", StringComparison.Ordinal))
            {
                string baseKey = kvp.Key[..^".weight".Length];
                if (scales.TryGetValue(baseKey, out Tensor? scaleT) && scaleT.DType == DType.F32)
                {
                    float scale = ((float*)scaleT.DataPointer)[0];
                    kvp.Value.Fp8ScaleFactor = scale;
                }
            }

            result[kvp.Key] = kvp.Value;
        }
        return result;
    }
}
