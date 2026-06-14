using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Converter for Zeta-Chroma (<c>lodestones/Zeta-Chroma</c>) pixel-proto single-file safetensors.
/// Zeta-Chroma is the Z-Image S3-DiT retrained for pixel space, so this is a thin wrapper over
/// <see cref="ZImageCheckpointConverter"/>: the shared partitioner already strips wrappers (incl. the
/// <c>_orig_mod.</c> torch.compile prefix), folds <c>fp8_scaled</c> companions, and buckets the Zeta-only
/// <c>dec_net.*</c> decoder-head keys into the transformer dict. The Diffusion-side
/// <c>ZetaChromaTransformer.LoadWeights</c> validates the decoder layout defensively.</summary>
public sealed class ZetaChromaCheckpointConverter
{
    /// <summary>Partitions a flat dict of Zeta-Chroma safetensors keys (delegates to the Z-Image partitioner).</summary>
    public static ZImageCheckpointConverter.ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
        => ZImageCheckpointConverter.Convert(allWeights);

    /// <summary>Loads and partitions a Zeta-Chroma single-file checkpoint.</summary>
    public static (ZImageCheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(
        string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ZImageCheckpointConverter.ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>True when a partitioned Z-Image-family transformer dict is a Zeta-Chroma pixel checkpoint
    /// (the <c>dec_net.*</c> decoder head replaces <c>final_layer.*</c>).</summary>
    public static bool IsZetaChroma(IReadOnlyDictionary<string, Tensor> transformerWeights)
    {
        foreach (string key in transformerWeights.Keys)
        {
            if (key.StartsWith("dec_net.", StringComparison.Ordinal))
                return true;
        }
        return false;
    }
}
