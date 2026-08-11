using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converter for Z-Image (Tongyi Lab) single-file safetensors checkpoints. The SwarmUI single-file FP8Mix format uses ComfyUI's <c>fp8_scaled</c> layout with per-tensor <c>.weight_scale</c> companions and <c>.comfy_quant</c> metadata blobs. The transformer key naming is the official Z-Image / Tongyi naming (NOT diffusers): <c>x_embedder.{weight,bias}</c>, <c>final_layer.{adaLN_modulation.1,linear}.*</c>, <c>cap_pad_token</c>, <c>x_pad_token</c>, <c>layers.{i}.attention.qkv.weight</c> (fused), <c>layers.{i}.attention.out.weight</c>, <c>layers.{i}.attention.{q_norm,k_norm}</c>. This converter is mostly passthrough — its job is to fold weight scales into <see cref="Tensor.Fp8ScaleFactor"/>, drop quantization metadata, and partition transformer/VAE/text-encoder buckets.</summary>
public sealed class ZImageCheckpointConverter
{
    /// <summary>Sampling variant. Base and Turbo have the same tensor architecture, so this cannot be inferred
    /// from weight shapes; single-file loaders resolve it from an explicit filename token.</summary>
    public enum CheckpointVariant
    {
        Unknown,
        Turbo,
        Base,
    }

    /// <summary>Result of partitioning a Z-Image single-file safetensors checkpoint.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Z-Image transformer weights — pass directly to <c>ZImageTransformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>VAE weights (Flux VAE; usually empty in transformer-only checkpoints like SwarmUI's FP8Mix).</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }

        /// <summary>Qwen3-4B text encoder weights (usually empty — text encoder ships separately).</summary>
        public required Dictionary<string, Tensor> TextEncoder { get; init; }

        /// <summary>True if any transformer linear weight is FP8 — pipeline should preload via the FP8 path.</summary>
        public required bool IsFp8Mix { get; init; }

        /// <summary>Base/Turbo sampling variant resolved by <see cref="LoadAndConvert"/>. Direct dictionary
        /// conversion has no file identity and therefore returns <see cref="CheckpointVariant.Unknown"/>.</summary>
        public CheckpointVariant Variant { get; init; }
    }

    /// <summary>Loads and partitions a Z-Image single-file checkpoint.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(checkpointPath);
            ConvertedWeights converted = Convert(loader.GetAllTensors(), DetectVariantFromFileName(checkpointPath));
            return (converted, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Partitions a flat dict of Z-Image safetensors keys.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights,
        CheckpointVariant variant = CheckpointVariant.Unknown)
    {
        // Step 1: fold per-tensor weight_scale companions into Fp8ScaleFactor on each FP8 weight,
        // drop the .weight_scale and .comfy_quant metadata keys.
        Dictionary<string, Tensor> dequanted = ApplyFp8WeightScales(allWeights);

        Dictionary<string, Tensor> transformer = new(dequanted.Count);
        Dictionary<string, Tensor> vae = new();
        Dictionary<string, Tensor> textEncoder = new();

        foreach (KeyValuePair<string, Tensor> kvp in dequanted)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            // Optional `model.diffusion_model.` / `transformer.` wrapper, plus the `_orig_mod.`
            // torch.compile artifact prefix (Lodestone pixel-proto dumps, same as Chroma Radiance).
            string transformerKey = key;
            if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                transformerKey = key["model.diffusion_model.".Length..];
            else if (key.StartsWith("transformer.", StringComparison.Ordinal))
                transformerKey = key["transformer.".Length..];
            if (transformerKey.StartsWith("_orig_mod.", StringComparison.Ordinal))
                transformerKey = transformerKey["_orig_mod.".Length..];

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
            Variant = variant,
        };
    }

    /// <summary>Detects the sampling variant from a standalone filename token. Official Base and Turbo weights are
    /// architecturally indistinguishable. Safetensors metadata is not reliable enough as the sole contract: the
    /// official BF16 Base file has no metadata, while the known FP8 repacks identify themselves in the filename.
    /// Only the filename (not parent directories) is inspected. The official Base release carries NO variant token
    /// — it ships under the bare family name (<c>Z-Image.safetensors</c>, <c>z_image_bf16.safetensors</c>) — so a
    /// name that starts with the family name and lacks a variant token IS the Base checkpoint, not ambiguous.
    /// Anything else without an unambiguous token remains Unknown.</summary>
    public static CheckpointVariant DetectVariantFromFileName(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        string fileName = Path.GetFileNameWithoutExtension(checkpointPath);
        string[] tokens = fileName.Split(['-', '_', '.', ' '], StringSplitOptions.RemoveEmptyEntries);
        bool hasBase = tokens.Any(token => token.Equals("base", StringComparison.OrdinalIgnoreCase));
        bool hasTurbo = tokens.Any(token => token.Equals("turbo", StringComparison.OrdinalIgnoreCase));
        if (hasBase && hasTurbo)
            return CheckpointVariant.Unknown;
        if (hasBase || hasTurbo)
            return hasBase ? CheckpointVariant.Base : CheckpointVariant.Turbo;
        bool officialFamilyName = tokens.Any(token => token.Equals("zimage", StringComparison.OrdinalIgnoreCase));
        for (int i = 0; !officialFamilyName && i < tokens.Length - 1; i++)
        {
            officialFamilyName = tokens[i].Equals("z", StringComparison.OrdinalIgnoreCase)
                && tokens[i + 1].Equals("image", StringComparison.OrdinalIgnoreCase);
        }
        return officialFamilyName ? CheckpointVariant.Base : CheckpointVariant.Unknown;
    }

    /// <summary>Auto-detects layer counts and FP8 status from a transformer dict.</summary>
    public static (int numLayers, int numRefinerLayers, int hidden, int ffnDim, bool isFp8Mix) DetectArchitecture(
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

        int hidden = 3840;
        int ffnDim = 10240;
        // Hidden is implied by the fused qkv shape: [3*hidden, hidden].
        if (transformerWeights.TryGetValue("layers.0.attention.qkv.weight", out Tensor? qkv))
            hidden = (int)qkv.Shape[1];
        if (transformerWeights.TryGetValue("layers.0.feed_forward.w1.weight", out Tensor? w1))
            ffnDim = (int)w1.Shape[0];

        int numRefiner = Math.Min(maxNoiseRefiner, maxContextRefiner) + 1;
        if (numRefiner <= 0) numRefiner = 2;

        return (maxLayer + 1, numRefiner, hidden, ffnDim, DetectFp8Mix(transformerWeights));
    }

    private static bool IsTransformerKey(string key)
    {
        return key.StartsWith("layers.", StringComparison.Ordinal)
            || key.StartsWith("noise_refiner.", StringComparison.Ordinal)
            || key.StartsWith("context_refiner.", StringComparison.Ordinal)
            || key.StartsWith("t_embedder.", StringComparison.Ordinal)
            || key.StartsWith("cap_embedder.", StringComparison.Ordinal)
            || key.StartsWith("x_embedder.", StringComparison.Ordinal)
            || key.StartsWith("final_layer.", StringComparison.Ordinal)
            // Zeta-Chroma pixel decoder head (absent on classic Z-Image; replaces final_layer.*).
            || key.StartsWith("dec_net.", StringComparison.Ordinal)
            || key == "cap_pad_token"
            || key == "x_pad_token";
    }

    private static bool DetectFp8Mix(IReadOnlyDictionary<string, Tensor> transformer)
    {
        if (transformer.TryGetValue("layers.0.attention.qkv.weight", out Tensor? probe))
            return probe.DType == DType.F8E4M3 || probe.DType == DType.F8E5M2;

        foreach (Tensor t in transformer.Values)
        {
            if (t.DType == DType.F8E4M3 || t.DType == DType.F8E5M2)
                return true;
        }
        return false;
    }

    /// <summary>Folds ComfyUI <c>fp8_scaled</c> per-tensor scale companions into <see cref="Tensor.Fp8ScaleFactor"/>. Z-Image uses the suffix <c>.weight_scale</c> (Flux uses <c>.scale_weight</c>; same idea, different naming). Also drops <c>.comfy_quant</c> metadata blobs (27-byte U8 tensors that describe the quantization config — purely informational).</summary>
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
            // Drop scale and quant-metadata companions.
            if (kvp.Key.EndsWith(".weight_scale", StringComparison.Ordinal) ||
                kvp.Key.EndsWith(".comfy_quant", StringComparison.Ordinal))
            {
                continue;
            }

            // For an FP8 weight tensor with a matching scale companion, attach the scale.
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
