using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads + buckets an OmniGen 2 safetensors checkpoint into transformer / VAE / text encoder
/// dictionaries. OmniGen 2 ships in canonical diffusers naming.</summary>
public sealed class OmniGen2CheckpointConverter
{
    /// <summary>Result of partitioning an OmniGen 2 safetensors file.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Transformer { get; init; }
        public required Dictionary<string, Tensor> Vae { get; init; }
        public required Dictionary<string, Tensor> TextEncoder { get; init; }
        public required bool IsFp8Mix { get; init; }
    }

    /// <summary>Loads and partitions an OmniGen 2 single-file checkpoint.</summary>
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
        Dictionary<string, Tensor> textEncoder = new();

        foreach (KeyValuePair<string, Tensor> kvp in dequanted)
        {
            string key = kvp.Key;
            Tensor tensor = kvp.Value;

            if (key.StartsWith("vae.", StringComparison.Ordinal))
            {
                vae[key] = tensor;
                continue;
            }
            if (key.StartsWith("text_encoder.", StringComparison.Ordinal) ||
                key.StartsWith("text_encoders.", StringComparison.Ordinal))
            {
                textEncoder[key] = tensor;
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
            TextEncoder = textEncoder,
            IsFp8Mix = isFp8Mix,
        };
    }
}
