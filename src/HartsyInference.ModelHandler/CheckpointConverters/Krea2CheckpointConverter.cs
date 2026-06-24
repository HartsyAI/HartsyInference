using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Loads Krea 2 (`krea/Krea-2-Turbo`, `CalamitousFelicitousness/Krea-2-Base-Diffusers`) checkpoints. Krea 2
/// ships in the diffusers folder layout: <c>{root}/transformer/</c> (the
/// <see cref="HartsyInference.Diffusion.Models.Denoisers.Krea2Transformer"/>), <c>{root}/text_encoder/</c> (Qwen3-VL-4B
/// language tower), <c>{root}/vae/</c> (the Qwen-Image VAE), plus <c>{root}/scheduler/</c> + <c>{root}/tokenizer/</c>.
///
/// <para>Transformer keys are diffusers-native (bare <c>img_in.weight</c>, <c>transformer_blocks.{i}.*</c>,
/// <c>text_fusion.*</c>, <c>time_embed.*</c>, <c>final_layer.*</c>) — only an optional <c>transformer.</c> /
/// <c>model.diffusion_model.</c> prefix is stripped and fp8_scaled companions are folded via
/// <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>. The Qwen3-VL-4B keys are remapped to the
/// <see cref="HartsyInference.Diffusion.Models.TextEncoders.LlamaStyleEncoder"/> convention (vision tower + lm_head
/// dropped). The Qwen-Image VAE keys are consumed directly.</para>
///
/// <para>TODO: the Comfy single-file release (fp8/nv4 under <c>diffusion_models/</c>) may use the original short-name
/// config (<c>features</c>/<c>heads</c>/… ) and raw weight keys; add a single-file key remap when that layout is needed.</para></summary>
public sealed class Krea2CheckpointConverter
{
    /// <summary>Loads the Krea 2 transformer from <c>{root}/transformer/</c> (or a <c>krea2*</c> single file). fp8 folded.</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTransformer(string rootPath)
    {
        string[] shards = DiscoverShards(Path.Combine(rootPath, "transformer"), rootPath, "krea2");
        return LoadShards(shards, 1200, StripTransformerPrefix);
    }

    /// <summary>Loads the Qwen-Image VAE from <c>{root}/vae/</c> (keys consumed directly by <c>QwenImageVaeDecoder</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadVae(string rootPath)
    {
        string dir = Path.Combine(rootPath, "vae");
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException($"VAE folder not found: {dir}");
        string[] shards = Directory.GetFiles(dir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No VAE .safetensors found under {dir}.");
        Array.Sort(shards);
        return LoadShards(shards, 400, k => k);
    }

    /// <summary>Loads + remaps the Qwen3-VL-4B language tower from <c>{root}/text_encoder/</c> to the
    /// <c>LlamaStyleEncoder</c> convention (drops the vision tower and <c>lm_head</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTextEncoder(string rootPath)
    {
        string te1 = Path.Combine(rootPath, "text_encoder");
        string te2 = Path.Combine(rootPath, "text_encoders");
        string dir = Directory.Exists(te1) ? te1 : te2;
        string[] shards = DiscoverShards(dir, rootPath, "qwen");
        return LoadShardsRemap(shards, 800, RemapQwenLanguageKey);
    }

    private static string[] DiscoverShards(string preferredDir, string rootPath, string what)
    {
        if (Directory.Exists(preferredDir))
        {
            string[] s = Directory.GetFiles(preferredDir, "*.safetensors");
            if (s.Length > 0) { Array.Sort(s); return s; }
        }
        string[] all = Directory.GetFiles(rootPath, "*.safetensors");
        string[] match = Array.FindAll(all, f => Path.GetFileName(f).ToLowerInvariant().Contains(what));
        if (match.Length == 0)
            throw new FileNotFoundException($"No Krea 2 {what} .safetensors found under {preferredDir} or {rootPath}.");
        Array.Sort(match);
        return match;
    }

    private static (Dictionary<string, Tensor>, IReadOnlyList<SafeTensorsLoader>) LoadShards(
        string[] shards, int capacity, Func<string, string> keyMap)
    {
        Dictionary<string, Tensor> merged = new(capacity);
        List<SafeTensorsLoader> loaders = new(shards.Length);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                {
                    if (kvp.Key.EndsWith(".scaled_fp8") || kvp.Key == "scaled_fp8") continue;
                    merged[keyMap(kvp.Key)] = kvp.Value;
                }
            }
            return (CheckpointConvertUtils.ApplyFp8ScaledDequant(merged), loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    private static (Dictionary<string, Tensor>, IReadOnlyList<SafeTensorsLoader>) LoadShardsRemap(
        string[] shards, int capacity, Func<string, string?> keyMap)
    {
        Dictionary<string, Tensor> merged = new(capacity);
        List<SafeTensorsLoader> loaders = new(shards.Length);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                {
                    if (kvp.Key.EndsWith(".scaled_fp8") || kvp.Key == "scaled_fp8") continue;
                    string? mapped = keyMap(kvp.Key);
                    if (mapped is not null) merged[mapped] = kvp.Value;
                }
            }
            return (CheckpointConvertUtils.ApplyFp8ScaledDequant(merged), loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    private static string StripTransformerPrefix(string key)
    {
        if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
            return key["model.diffusion_model.".Length..];
        if (key.StartsWith("diffusion_model.", StringComparison.Ordinal))
            return key["diffusion_model.".Length..];
        if (key.StartsWith("transformer.", StringComparison.Ordinal))
            return key["transformer.".Length..];
        return key;
    }

    private static string? RemapQwenLanguageKey(string key)
    {
        if (key.Contains(".visual.") || key.StartsWith("visual.", StringComparison.Ordinal)) return null;
        if (key.Contains("lm_head")) return null;

        int lm = key.LastIndexOf("language_model.", StringComparison.Ordinal);
        string suffix = lm >= 0 ? key[(lm + "language_model.".Length)..] : key;
        if (suffix.StartsWith("model.", StringComparison.Ordinal))
            suffix = suffix["model.".Length..];

        if (suffix.StartsWith("layers.", StringComparison.Ordinal)
            || suffix.StartsWith("embed_tokens.", StringComparison.Ordinal)
            || suffix == "norm.weight")
        {
            return "model." + suffix;
        }
        return null;
    }
}
