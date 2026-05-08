using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Loads + buckets a HiDream i1 single-file safetensors checkpoint into transformer / VAE /
/// CLIP-L / CLIP-G / T5-XXL / Llama-3.1 dictionaries. HiDream ships in canonical diffusers naming, so no
/// key remapping is required. FP8 <c>.scale_weight</c> companion tensors are folded into
/// <see cref="Tensor.Fp8ScaleFactor"/> via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>.</summary>
public sealed class HiDreamCheckpointConverter
{
    /// <summary>Result of partitioning a HiDream i1 single-file checkpoint.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Transformer { get; init; }
        public required Dictionary<string, Tensor> Vae { get; init; }
        public required Dictionary<string, Tensor> ClipL { get; init; }
        public required Dictionary<string, Tensor> ClipG { get; init; }
        public required Dictionary<string, Tensor> T5 { get; init; }
        public required Dictionary<string, Tensor> Llama { get; init; }
        public required bool IsFp8Mix { get; init; }
    }

    /// <summary>Loads and partitions a HiDream i1 single-file checkpoint.</summary>
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
        Dictionary<string, Tensor> clipG = new();
        Dictionary<string, Tensor> t5 = new();
        Dictionary<string, Tensor> llama = new();

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
                key.StartsWith("text_encoders.clip_g.", StringComparison.Ordinal) ||
                key.StartsWith("conditioner.embedders.1.", StringComparison.Ordinal))
            {
                clipG[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder_3.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.t5xxl.", StringComparison.Ordinal) ||
                key.StartsWith("conditioner.embedders.2.", StringComparison.Ordinal))
            {
                t5[key] = tensor;
                continue;
            }

            if (key.StartsWith("text_encoder_4.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.llama.", StringComparison.Ordinal) ||
                key.StartsWith("conditioner.embedders.3.", StringComparison.Ordinal) ||
                key.StartsWith("llama.", StringComparison.Ordinal))
            {
                llama[key] = tensor;
                continue;
            }

            string transformerKey = key;
            if (transformerKey.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
                transformerKey = transformerKey["model.diffusion_model.".Length..];
            else if (transformerKey.StartsWith("transformer.", StringComparison.Ordinal))
                transformerKey = transformerKey["transformer.".Length..];

            transformer[transformerKey] = tensor;
        }

        bool isFp8Mix = false;
        foreach (Tensor t in transformer.Values)
        {
            if (t.DType == DType.F8E4M3 || t.DType == DType.F8E5M2) { isFp8Mix = true; break; }
        }

        return new ConvertedWeights
        {
            Transformer = transformer,
            Vae = vae,
            ClipL = clipL,
            ClipG = clipG,
            T5 = t5,
            Llama = llama,
            IsFp8Mix = isFp8Mix,
        };
    }
}
