using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Loads <c>kandinskylab/Kandinsky-5.0-T2I-Lite-*-Diffusers</c> safetensors into a
/// dictionary already keyed by the diffusers state-dict names that
/// <see cref="HartsyInference.Diffusion.Models.Denoisers.Kandinsky5Transformer"/> expects.
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
/// No sub-key renaming happens because the diffusers naming and the HartsyInference
/// <c>Kandinsky5Transformer.LoadWeights</c> contract are intentionally aligned 1:1.</summary>
public sealed class Kandinsky5CheckpointConverter
{
    private static readonly string[] _stripPrefixes = ["transformer.", "model."];

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
            for (int i = 0; i < _stripPrefixes.Length; i++)
            {
                if (trKey.StartsWith(_stripPrefixes[i], StringComparison.Ordinal))
                {
                    trKey = trKey[_stripPrefixes[i].Length..];
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

    /// <summary>Loads the T2V video transformer from either a diffusers <c>transformer/</c> directory or
    /// a single repackaged safetensors file. The video state dict uses the same canonical key set as T2I
    /// (plus the wider 33-channel <c>visual_embeddings.in_layer</c>), so the conversion path is shared.</summary>
    public static (ConvertedWeights weights, List<SafeTensorsLoader> loaders) LoadVideoTransformer(string transformerPathOrDir)
    {
        if (Directory.Exists(transformerPathOrDir))
            return LoadDiffusersFolder(transformerPathOrDir);

        if (!File.Exists(transformerPathOrDir))
            throw new FileNotFoundException($"Kandinsky 5 video transformer not found: {transformerPathOrDir}");

        (ConvertedWeights weights, SafeTensorsLoader loader) = LoadAndConvert(transformerPathOrDir);
        return (weights, [loader]);
    }

    /// <summary>Loads the HunyuanVideo VAE shard (<c>vae/diffusion_pytorch_model.safetensors</c> or a
    /// directory containing it) for <c>HunyuanVideoVaeDecoder</c>/<c>HunyuanVideoVaeEncoder</c>. The
    /// diffusers shard is already keyed <c>encoder.* / decoder.* / quant_conv.* / post_quant_conv.*</c>;
    /// a <c>vae.</c> wrapper prefix (single-file repacks) is stripped. Caller disposes the loaders.</summary>
    public static (Dictionary<string, Tensor> weights, List<SafeTensorsLoader> loaders) LoadHunyuanVideoVae(string vaePathOrDir)
    {
        string[] shards;
        if (Directory.Exists(vaePathOrDir))
        {
            shards = Directory.GetFiles(vaePathOrDir, "*.safetensors");
            Array.Sort(shards, StringComparer.Ordinal);
            if (shards.Length == 0)
                throw new FileNotFoundException($"No safetensors found in Kandinsky 5 VAE dir: {vaePathOrDir}");
        }
        else if (File.Exists(vaePathOrDir))
        {
            shards = [vaePathOrDir];
        }
        else
        {
            throw new FileNotFoundException($"Kandinsky 5 VAE not found: {vaePathOrDir}");
        }

        List<SafeTensorsLoader> loaders = new();
        Dictionary<string, Tensor> weights = new(1024);
        foreach (string shard in shards)
        {
            SafeTensorsLoader loader = new();
            loader.Load(shard);
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            {
                string key = kvp.Key;
                if (key.StartsWith("vae.", StringComparison.Ordinal))
                    key = key["vae.".Length..];
                weights[key] = kvp.Value;
            }
            loaders.Add(loader);
        }
        return (weights, loaders);
    }
}
