using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Loads <c>kandinskylab/Kandinsky-5.0-T2I-Lite-*-Diffusers</c> safetensors into a
/// dictionary already keyed by the diffusers state-dict names that
/// <see cref="SharpInference.Diffusion.Models.Denoisers.Kandinsky5Transformer"/> expects.
///
/// The diffusers folder layout ships the transformer in a per-component <c>transformer/</c>
/// subdirectory whose state dict is already in the canonical form
/// (<c>text_embeddings.in_layer.weight</c>, <c>visual_transformer_blocks.{i}.self_attention.to_query.weight</c>,
/// etc.). This converter therefore mostly handles two real-world quirks:
/// <list type="bullet">
/// <item>Single-file repackaged checkpoints sometimes prepend a <c>transformer.</c> or <c>model.</c>
/// prefix to every key. We strip the first matching prefix.</item>
/// <item>ComfyUI-style FP8 scaled-weight companions (<c>*.scale_weight</c>) are folded into
/// <see cref="Tensor.Fp8ScaleFactor"/> via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>.</item>
/// </list>
///
/// No sub-key renaming happens because the diffusers naming and the SharpInference
/// <c>Kandinsky5Transformer.LoadWeights</c> contract are intentionally aligned 1:1.</summary>
public sealed class Kandinsky5CheckpointConverter
{
    private static readonly string[] StripPrefixes = ["transformer.", "model."];

    /// <summary>Result bundle. The Lite t2i model ships transformer + (CLIP + Qwen2.5-VL) text encoders
    /// + Flux VAE in separate sub-directories of the diffusers repo, so when given a single transformer
    /// safetensors only <see cref="Transformer"/> is populated.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Transformer { get; init; }
    }

    /// <summary>Converts a flat weight dictionary loaded from a Kandinsky 5 transformer safetensors.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(allWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string key = kvp.Key;
            if (key.EndsWith(".scaled_fp8") || key == "scaled_fp8")
                continue;

            string trKey = key;
            for (int i = 0; i < StripPrefixes.Length; i++)
            {
                if (trKey.StartsWith(StripPrefixes[i], StringComparison.Ordinal))
                {
                    trKey = trKey[StripPrefixes[i].Length..];
                    break;
                }
            }

            transformer[trKey] = kvp.Value;
        }

        return new ConvertedWeights { Transformer = transformer };
    }

    /// <summary>Convenience: load a single safetensors file and convert. The caller owns the loader
    /// and must dispose it once weights are no longer referenced.</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        ConvertedWeights converted = Convert(raw);
        return (converted, loader);
    }

    /// <summary>Loads a diffusers folder layout: scans the <c>transformer/</c> subdirectory for
    /// safetensors shards and merges them into a single dictionary. Returns the converted result
    /// plus the loaders (disposed by the caller).</summary>
    public static (ConvertedWeights weights, List<SafeTensorsLoader> loaders) LoadDiffusersFolder(string transformerDir)
    {
        if (!Directory.Exists(transformerDir))
            throw new DirectoryNotFoundException(
                $"Kandinsky 5 transformer dir not found: {transformerDir}");

        List<SafeTensorsLoader> loaders = new();
        Dictionary<string, Tensor> merged = new(2048);
        string[] shards = Directory.GetFiles(transformerDir, "*.safetensors");
        Array.Sort(shards, StringComparer.Ordinal);
        foreach (string shard in shards)
        {
            SafeTensorsLoader loader = new();
            loader.Load(shard);
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                merged[kvp.Key] = kvp.Value;
            loaders.Add(loader);
        }

        ConvertedWeights converted = Convert(merged);
        return (converted, loaders);
    }
}
