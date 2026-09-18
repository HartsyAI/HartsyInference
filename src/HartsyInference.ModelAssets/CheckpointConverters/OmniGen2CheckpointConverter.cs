using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads + buckets an OmniGen 2 safetensors checkpoint into transformer / VAE / text encoder dictionaries. OmniGen 2 ships in canonical diffusers naming.</summary>
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

    /// <summary>Partitions a flat dict by key prefix.</summary>
    /// <remarks>Quantization companions are expected to be folded already — <see cref="Checkpoints.CheckpointSource"/>
    /// does it before any converter runs, because folding after a converter has stripped a key prefix pairs nothing
    /// and drops the scale silently.</remarks>
    public static ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> allWeights)
    {
        CheckpointConvertUtils.RequireFoldedCompanions(allWeights, nameof(OmniGen2CheckpointConverter));

        Dictionary<string, Tensor> transformer = new();
        Dictionary<string, Tensor> vae = new();
        Dictionary<string, Tensor> textEncoder = new();

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
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
