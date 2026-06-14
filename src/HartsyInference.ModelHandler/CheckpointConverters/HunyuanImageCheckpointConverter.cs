using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Loads + buckets a Hunyuan Image 2.1 single-file safetensors checkpoint into transformer / VAE
/// / CLIP / T5 dictionaries. The diffusers naming convention is preserved verbatim — Hunyuan ships in
/// canonical diffusers layout, so no key remapping is required. FP8 <c>.scale_weight</c> companion
/// tensors are folded into <see cref="Tensor.Fp8ScaleFactor"/> via the shared
/// <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/> helper.</summary>
public sealed class HunyuanImageCheckpointConverter
{
    /// <summary>Result of partitioning a Hunyuan Image 2.1 safetensors file.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Hunyuan Image transformer weights — pass to <c>HunyuanImageTransformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>32-channel Hunyuan VAE weights.</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }

        /// <summary>CLIP-L text encoder weights (when bundled).</summary>
        public required Dictionary<string, Tensor> ClipL { get; init; }

        /// <summary>T5-XXL text encoder weights (when bundled).</summary>
        public required Dictionary<string, Tensor> T5 { get; init; }

        /// <summary>True when at least one transformer linear has FP8 storage.</summary>
        public required bool IsFp8Mix { get; init; }
    }

    /// <summary>Loads and partitions a Hunyuan Image 2.1 single-file checkpoint.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>Partitions a flat dict by key prefix.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> dequanted = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new();
        Dictionary<string, Tensor> vae = new();
        Dictionary<string, Tensor> clipL = new();
        Dictionary<string, Tensor> t5 = new();

        foreach (KeyValuePair<string, Tensor> kvp in dequanted)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.StartsWith("vae.", StringComparison.Ordinal) ||
                key.StartsWith("first_stage_model.", StringComparison.Ordinal))
            {
                vae[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.clip_l.", StringComparison.Ordinal) ||
                key.StartsWith("conditioner.embedders.0.", StringComparison.Ordinal))
            {
                clipL[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder_2.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.t5xxl.", StringComparison.Ordinal) ||
                key.StartsWith("conditioner.embedders.1.", StringComparison.Ordinal))
            {
                t5[key] = tensor;
                continue;
            }

            string transformerKey = key;
            if (transformerKey.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                transformerKey = transformerKey["model.diffusion_model.".Length..];
            else if (transformerKey.StartsWith("transformer.", StringComparison.Ordinal))
                transformerKey = transformerKey["transformer.".Length..];

            transformer[transformerKey] = tensor;
        }

        bool isFp8Mix = DetectFp8Mix(transformer);

        return new ConvertedWeights
        {
            Transformer = transformer,
            Vae = vae,
            ClipL = clipL,
            T5 = t5,
            IsFp8Mix = isFp8Mix,
        };
    }

    private static bool DetectFp8Mix(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values)
        {
            if (t.DType == DType.F8E4M3 || t.DType == DType.F8E5M2) return true;
        }
        return false;
    }
}
