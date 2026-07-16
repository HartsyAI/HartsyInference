using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Loads Lance (ByteDance, Apache-2.0) checkpoints. Lance ships <c>{variant}/model.safetensors</c> (the MoT backbone + generation heads), a frozen <c>Qwen2.5-VL-ViT/vit.safetensors</c> (editing/understanding only, separate file), and <c>Wan2.2_VAE.pth</c> (the 3D causal VAE — a PyTorch <c>.pth</c>, NOT safetensors).
///
/// <para>Real checkpoint key families (confirmed against <c>Lance_3B/model.safetensors</c>, 2026-07): <c>language_model.model.layers.{i}.*</c> with <c>*_moe_gen</c> siblings (incl. per-head <c>q_norm</c>/<c>k_norm</c>), <c>language_model.model.{embed_tokens,norm,norm_moe_gen}</c>, <c>language_model.lm_head</c> (unused by generation), plus top-level heads <c>vae2llm.*</c> (Linear 48→2048), <c>llm2vae.*</c> (2048→48), <c>latent_pos_embed.pos_embed</c> (frozen sin-cos [4096,2048]), <c>time_embedder.mlp.{0,2}.*</c>. This converter strips the <c>language_model.model.</c> prefix off backbone keys and passes the top-level heads through unchanged; <c>lm_head</c> is dropped.</para>
///
/// <para><b>VAE note:</b> <c>Wan2.2_VAE.pth</c> must be converted to safetensors offline (the <see cref="SafeTensorsLoader"/> can't parse <c>.pth</c>); point <see cref="LoadVae"/> at the converted file (the same file the Wan2.2 video loader uses).</para></summary>
public sealed class LanceCheckpointConverter
{
    /// <summary>Result buckets from <c>model.safetensors</c>.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Backbone + generation-head weights for <see cref="HartsyInference.Diffusion.Models.Denoisers.LanceTransformer"/>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>ViT + connector weights (editing/understanding path). Empty for a pure T2I image checkpoint or when not needed.</summary>
        public required Dictionary<string, Tensor> Vit { get; init; }
    }

    /// <summary>Loads + buckets <c>model.safetensors</c> from <paramref name="variantDir"/> (e.g. <c>{root}/Lance_3B</c> or <c>{root}/Lance_3B_Video</c>).</summary>
    public static (ConvertedWeights Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadAndConvert(string variantDir)
    {
        if (!Directory.Exists(variantDir))
            throw new DirectoryNotFoundException($"Lance variant folder not found: {variantDir}");
        string[] shards = Directory.GetFiles(variantDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No model.safetensors shards found in: {variantDir}");
        Array.Sort(shards);

        Dictionary<string, Tensor> transformer = new(4000);
        Dictionary<string, Tensor> vit = new(800);
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
                    Bucket(kvp.Key, kvp.Value, transformer, vit);
                }
            }
            Dictionary<string, Tensor> foldedT = CheckpointConvertUtils.ApplyFp8ScaledDequant(transformer);
            ConvertedWeights converted = new() { Transformer = foldedT, Vit = vit };
            return (converted, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    /// <summary>Loads the Wan2.2 VAE weights from a safetensors file/folder (convert <c>Wan2.2_VAE.pth</c> → safetensors offline first). Strips a leading <c>model.</c> so keys match <c>Wan22VaeDecoder</c> (<c>conv2.*</c>, <c>decoder.*</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadVae(string vaePathOrDir)
    {
        string[] shards = Directory.Exists(vaePathOrDir)
            ? Directory.GetFiles(vaePathOrDir, "*.safetensors")
            : [vaePathOrDir];
        if (shards.Length == 0 || !File.Exists(shards[0]))
            throw new FileNotFoundException($"Wan2.2 VAE safetensors not found at: {vaePathOrDir} (convert Wan2.2_VAE.pth → safetensors first).");
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
                {
                    string key = kvp.Key.StartsWith("model.", StringComparison.Ordinal) ? kvp.Key["model.".Length..] : kvp.Key;
                    merged[key] = kvp.Value;
                }
            }
            return (merged, loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }

    private static void Bucket(string key, Tensor value, Dictionary<string, Tensor> transformer, Dictionary<string, Tensor> vit)
    {
        (LanceBucket bucket, string? mapped) = RouteKey(key);
        switch (bucket)
        {
            case LanceBucket.Transformer: transformer[mapped!] = value; break;
            case LanceBucket.Vit: vit[mapped!] = value; break;
            case LanceBucket.Drop: break;
        }
    }

    /// <summary>Destination bucket for a checkpoint key.</summary>
    public enum LanceBucket
    {
        /// <summary>Backbone + generation heads (consumed by <c>LanceTransformer</c>).</summary>
        Transformer,
        /// <summary>ViT + connector (editing/understanding path).</summary>
        Vit,
        /// <summary>Not used by the T2I path.</summary>
        Drop,
    }

    /// <summary>Pure key-routing logic (testable without files): returns the destination bucket and the mapped key (prefix-stripped for the backbone). Returns <c>(Drop, null)</c> for keys the T2I path doesn't consume.</summary>
    public static (LanceBucket Bucket, string? MappedKey) RouteKey(string key)
    {
        if (key.StartsWith("vit.", StringComparison.Ordinal) || key.StartsWith("connector.", StringComparison.Ordinal))
            return (LanceBucket.Vit, key);

        if (key.Contains("lm_head") || key.StartsWith("task_embed", StringComparison.Ordinal)
            || key.StartsWith("modality_embed", StringComparison.Ordinal))
            return (LanceBucket.Drop, null);

        if (key.StartsWith("language_model.model.", StringComparison.Ordinal))
            return (LanceBucket.Transformer, key["language_model.model.".Length..]);
        if (key.StartsWith("language_model.", StringComparison.Ordinal))
            return (LanceBucket.Transformer, key["language_model.".Length..]);

        return (LanceBucket.Transformer, key);   // top-level heads: vae2llm, llm2vae, latent_pos_embed, time_embedder.*
    }
}
