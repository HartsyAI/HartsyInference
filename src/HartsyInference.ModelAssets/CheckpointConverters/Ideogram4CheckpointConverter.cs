using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads Ideogram 4 (<c>ideogram-oss/ideogram4</c>) checkpoints. Ideogram 4 ships **two transformers** of identical architecture but different weights — a conditional one (<c>transformer/</c>) and an unconditional one (<c>unconditional_transformer/</c>) used for the asymmetric-CFG negative pass — plus a Qwen3-VL-8B text encoder and the Flux.2 VAE.
///
/// Accepts both layouts:
/// <list type="bullet">
///   <item><b>diffusers / official</b>: <c>{root}/transformer/</c>, <c>{root}/unconditional_transformer/</c>, <c>{root}/text_encoder/</c>, <c>{root}/vae/</c> (each a sharded folder).</item>
///   <item><b>Comfy-Org</b>: <c>{root}/diffusion_models/ideogram4_*scaled.safetensors</c> (+ <c>*_unconditional_*</c>), <c>{root}/text_encoders/qwen3vl_8b_*.safetensors</c>, <c>{root}/vae/flux2-vae.safetensors</c>.</item>
/// </list>
///
/// Transformer keys are diffusers-native (bare <c>input_proj.weight</c>, <c>layers.{i}.*</c>) — only an optional <c>transformer.</c>/<c>model.diffusion_model.</c> prefix is stripped, and fp8_scaled <c>*.scale_weight</c> companions are folded into <see cref="Tensor.Fp8ScaleFactor"/>. The Qwen3-VL keys are remapped to the <see cref="HartsyInference.Diffusion.Models.TextEncoders.LlamaStyleEncoder"/> convention (<c>model.embed_tokens</c>, <c>model.layers.{i}.*</c>, <c>model.norm</c>); the vision tower and <c>lm_head</c> are dropped.</summary>
public sealed class Ideogram4CheckpointConverter
{
    /// <summary>Loads one transformer (conditional or unconditional). Tries, in order: the named folder, the diffusers subfolder, then a Comfy single file matching <c>ideogram4*</c> (with/without <c>unconditional</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTransformer(
        string rootPath, bool unconditional)
    {
        string folderName = unconditional ? "unconditional_transformer" : "transformer";
        string diffusersDir = Path.Combine(rootPath, folderName);

        string[] shards;
        if (Directory.Exists(diffusersDir))
        {
            shards = Directory.GetFiles(diffusersDir, "*.safetensors");
        }
        else
        {
            // Comfy single-file layout under diffusion_models/.
            string comfyDir = Path.Combine(rootPath, "diffusion_models");
            string searchDir = Directory.Exists(comfyDir) ? comfyDir : rootPath;
            string[] all = Directory.GetFiles(searchDir, "*.safetensors");
            shards = Array.FindAll(all, f =>
            {
                string n = Path.GetFileName(f).ToLowerInvariant();
                bool isIdeogram = n.Contains("ideogram4") || n.Contains("ideogram-4") || n.Contains("ideogram_4");
                bool isUncond = n.Contains("unconditional") || n.Contains("uncond");
                return isIdeogram && (unconditional ? isUncond : !isUncond);
            });
        }

        if (shards.Length == 0)
            throw new FileNotFoundException($"No Ideogram 4 {(unconditional ? "unconditional " : "")}transformer .safetensors found under {rootPath}.");
        Array.Sort(shards);

        Dictionary<string, Tensor> merged = new(2000);
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
                    merged[StripTransformerPrefix(kvp.Key)] = kvp.Value;
                }
            }
            Dictionary<string, Tensor> folded = CheckpointConvertUtils.ApplyFp8ScaledDequant(merged);
            return (folded, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    /// <summary>Loads + remaps the Qwen3-VL-8B text encoder to the <c>LlamaStyleEncoder</c> key convention. Drops the vision tower and <c>lm_head</c>.</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTextEncoder(string rootPath)
    {
        string te1 = Path.Combine(rootPath, "text_encoder");
        string te2 = Path.Combine(rootPath, "text_encoders");
        string teDir = Directory.Exists(te1) ? te1 : te2;
        string[] shards;
        if (Directory.Exists(teDir))
        {
            string[] all = Directory.GetFiles(teDir, "*.safetensors");
            // Prefer a qwen3vl file when several encoders share the folder (Comfy bundles gemma4 too).
            string[] qwen = Array.FindAll(all, f => Path.GetFileName(f).ToLowerInvariant().Contains("qwen"));
            shards = qwen.Length > 0 ? qwen : all;
        }
        else
        {
            throw new DirectoryNotFoundException($"text_encoder folder not found: {te1} or {te2}");
        }
        if (shards.Length == 0)
            throw new FileNotFoundException($"No Qwen3-VL text-encoder .safetensors found under {teDir}.");
        Array.Sort(shards);

        Dictionary<string, Tensor> merged = new(800);
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
                    string? mapped = RemapQwenKey(kvp.Key);
                    if (mapped is not null)
                        merged[mapped] = kvp.Value;
                }
            }
            Dictionary<string, Tensor> folded = CheckpointConvertUtils.ApplyFp8ScaledDequant(merged);
            return (folded, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    /// <summary>Loads VAE shards from <c>{root}/vae/</c> (diffusers Flux.2 VAE keys, consumed directly by <c>VaeDecoder</c> + <c>VaeConfig.Flux2</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadVae(string rootPath)
    {
        string vaeDir = Path.Combine(rootPath, "vae");
        if (!Directory.Exists(vaeDir))
            throw new DirectoryNotFoundException($"VAE folder not found: {vaeDir}");
        string[] shards = Directory.GetFiles(vaeDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No VAE .safetensors found under {vaeDir}.");
        Array.Sort(shards);

        Dictionary<string, Tensor> merged = new(400);
        List<SafeTensorsLoader> loaders = new(shards.Length);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                    merged[kvp.Key] = kvp.Value;
            }
            return (merged, loaders);
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

    /// <summary>Maps a Qwen3-VL checkpoint key to the <c>LlamaStyleEncoder</c> convention, or returns null to drop it (vision tower, <c>lm_head</c>, non-text heads). The language tower lives under <c>language_model.</c> in HF Qwen3-VL; we re-root it at <c>model.</c>.</summary>
    private static string? RemapQwenKey(string key)
    {
        // Drop the vision tower and any non-text generation heads.
        if (key.Contains(".visual.") || key.StartsWith("visual.", StringComparison.Ordinal)) return null;
        if (key.Contains("lm_head")) return null;

        // Find the language-tower suffix (everything after the last "language_model.").
        int lm = key.LastIndexOf("language_model.", StringComparison.Ordinal);
        string suffix = lm >= 0 ? key[(lm + "language_model.".Length)..] : key;

        // Strip any residual "model." so we can re-root uniformly.
        if (suffix.StartsWith("model.", StringComparison.Ordinal))
            suffix = suffix["model.".Length..];

        // Keep only the encoder-relevant trees.
        if (suffix.StartsWith("layers.", StringComparison.Ordinal)
            || suffix.StartsWith("embed_tokens.", StringComparison.Ordinal)
            || suffix == "norm.weight")
        {
            return "model." + suffix;
        }
        return null;
    }
}
