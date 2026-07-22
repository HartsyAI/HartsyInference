using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads the Oasis-500m checkpoints (<c>Etched/oasis-500m</c>, MIT): <c>oasis500m.safetensors</c> (the
/// DiT-S/2) and <c>vit-l-20.safetensors</c> (the continuous ViT-VAE). Both ship as plain PyTorch state dicts whose
/// keys match the HartsyInference loaders 1:1 (inferred — confirmed at first key dump), so conversion is a prefix
/// strip (<c>model.</c>/<c>module.</c> wrappers seen in community forks) plus dropping non-persistent
/// <c>rotary_freqs</c> buffers if a repack included them (HartsyInference recomputes RoPE at load).</summary>
public sealed class OasisCheckpointConverter
{
    private static readonly string[] _stripPrefixes = ["model.", "module."];

    /// <summary>Pure key mapping: strips wrapper prefixes; <c>null</c> drops recomputed RoPE buffers.</summary>
    public static string? MapKey(string key)
    {
        foreach (string prefix in _stripPrefixes)
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                key = key[prefix.Length..];
                break;
            }
        if (key.Contains("rotary_freqs", StringComparison.Ordinal))
            return null;
        return key;
    }

    /// <summary>Converts a flat state dict (works for both the DiT and the VAE file).</summary>
    public static Dictionary<string, Tensor> Convert(Dictionary<string, Tensor> allWeights)
    {
        Dictionary<string, Tensor> mapped = new(allWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string? key = MapKey(kvp.Key);
            if (key is not null) mapped[key] = kvp.Value;
        }
        return mapped;
    }

    /// <summary>Loads one safetensors file (DiT or VAE) and converts. The caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"Oasis checkpoint not found: {checkpointPath}");
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        return (Convert(loader.GetAllTensors()), loader);
    }
}
