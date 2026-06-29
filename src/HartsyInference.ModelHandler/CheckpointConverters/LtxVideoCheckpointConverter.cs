using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

/// <summary>Loads LTX-Video checkpoints into the diffusers state-dict names that
/// <see cref="HartsyInference.Diffusion.Models.Denoisers.LtxVideoTransformer"/> and
/// <see cref="HartsyInference.Diffusion.Models.Vae.LtxVideoVaeDecoder"/> expect.
///
/// <para>Two layouts: (1) the <c>Lightricks/LTX-Video</c> single-file checkpoints (e.g. <c>ltx-video-2b-v0.9.safetensors</c>),
/// which bundle the DiT under <c>model.diffusion_model.*</c> and the VAE under <c>vae.*</c> in the original naming —
/// keys are routed by prefix and renamed via the rule tables ported verbatim from diffusers
/// <c>scripts/convert_ltx_to_diffusers.py</c> (transformer: <c>patchify_proj→proj_in</c>, <c>adaln_single→time_embed</c>,
/// <c>q_norm/k_norm→norm_q/norm_k</c>; VAE: the ordered flat <c>up_blocks.{0..9}</c>/<c>down_blocks.{0..9}</c> regrouping +
/// <c>res_blocks→resnets</c> + <c>per_channel_statistics</c> → <c>latents_mean</c>/<c>latents_std</c>); (2) the diffusers repo
/// folder layout (<c>transformer/</c> + <c>vae/</c> shards, already diffusers-named) — pass-through.</para>
///
/// <para>The VAE rename table targets the released 0.9 base VAE (the variant <c>LtxVideoVaeDecoder</c> implements). The
/// 0.9.1+ timestep-conditioned decoder additionally renames <c>last_time_embedder→time_embedder</c> /
/// <c>last_scale_shift_table→scale_shift_table</c> (included). The per-block <c>decoder.{mid_block,up_blocks.N}.time_embedder.*</c>
    /// and decoder-level <c>decoder.time_embedder.*</c>/<c>decoder.scale_shift_table</c> keys pass through to the now
    /// timestep-conditioned <c>LtxVideoVaeDecoder</c> (consumed when built with <c>timestepConditioned: true</c>);
    /// numerics validation-pending vs diffusers LTXPipeline 0.9.7.</para></summary>
public sealed class LtxVideoCheckpointConverter
{
    private const string TransformerPrefix = "model.diffusion_model.";
    private const string VaePrefix = "vae.";

    private static readonly (string From, string To)[] _transformerRenames =
    [
        ("patchify_proj", "proj_in"),
        ("adaln_single", "time_embed"),
        ("q_norm", "norm_q"),
        ("k_norm", "norm_k"),
    ];

    // Ordered — applied sequentially per key like the python script's replace chain; later sources never match
    // earlier outputs. Ported verbatim from diffusers convert_ltx_to_diffusers.py VAE_KEYS_RENAME_DICT.
    private static readonly (string From, string To)[] _vaeRenames =
    [
        ("up_blocks.0", "mid_block"),
        ("up_blocks.1", "up_blocks.0"),
        ("up_blocks.2", "up_blocks.1.upsamplers.0"),
        ("up_blocks.3", "up_blocks.1"),
        ("up_blocks.4", "up_blocks.2.conv_in"),
        ("up_blocks.5", "up_blocks.2.upsamplers.0"),
        ("up_blocks.6", "up_blocks.2"),
        ("up_blocks.7", "up_blocks.3.conv_in"),
        ("up_blocks.8", "up_blocks.3.upsamplers.0"),
        ("up_blocks.9", "up_blocks.3"),
        ("down_blocks.1", "down_blocks.0.downsamplers.0"),
        ("down_blocks.2", "down_blocks.0.conv_out"),
        ("down_blocks.3", "down_blocks.1"),
        ("down_blocks.4", "down_blocks.1.downsamplers.0"),
        ("down_blocks.5", "down_blocks.1.conv_out"),
        ("down_blocks.6", "down_blocks.2"),
        ("down_blocks.7", "down_blocks.2.downsamplers.0"),
        ("down_blocks.8", "down_blocks.3"),
        ("down_blocks.9", "mid_block"),
        ("conv_shortcut", "conv_shortcut.conv"),
        ("res_blocks", "resnets"),
        ("norm3.norm", "norm3"),
        ("per_channel_statistics.mean-of-means", "latents_mean"),
        ("per_channel_statistics.std-of-means", "latents_std"),
        ("last_time_embedder", "time_embedder"),
        ("last_scale_shift_table", "scale_shift_table"),
    ];

    /// <summary>Destination bucket for a checkpoint key.</summary>
    public enum LtxBucket
    {
        /// <summary>DiT weights (consumed by <c>LtxVideoTransformer</c>).</summary>
        Transformer,
        /// <summary>VAE weights, <c>decoder.*</c>/<c>encoder.*</c> + <c>latents_mean</c>/<c>latents_std</c>.</summary>
        Vae,
        /// <summary>Unused metadata (per-channel stats leftovers, FP8 scale companions).</summary>
        Drop,
    }

    /// <summary>Result buckets. The T5-XXL text encoder ships separately (not bundled in LTX checkpoints).</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>DiT weights in diffusers naming for <c>LtxVideoTransformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>VAE weights in diffusers naming for <c>LtxVideoVaeDecoder.LoadWeights</c> (decoder keys; encoder keys
        /// are carried for a future encoder, <c>latents_mean</c>/<c>latents_std</c> for pipeline-side normalization).</summary>
        public required Dictionary<string, Tensor> Vae { get; init; }
    }

    /// <summary>True when the (prefix-stripped) VAE key set uses original LTX naming rather than diffusers naming.</summary>
    public static bool IsOriginalVaeNaming(IEnumerable<string> keys)
    {
        foreach (string raw in keys)
        {
            string key = raw.StartsWith(VaePrefix, StringComparison.Ordinal) ? raw[VaePrefix.Length..] : raw;
            if (key.Contains("res_blocks") || key.Contains("per_channel_statistics"))
                return true;
            if (key.Contains("resnets"))
                return false;
        }
        return false;
    }

    /// <summary>Pure key routing (testable without files): returns the destination bucket and the mapped key.
    /// Single-file keys carry the <c>model.diffusion_model.</c>/<c>vae.</c> prefixes; bare keys (diffusers folder
    /// shards) are treated as transformer or VAE by content. <paramref name="vaeOriginalNaming"/> gates the VAE
    /// regrouping renames (never apply them to already-diffusers-named keys — they would corrupt the layout).</summary>
    public static (LtxBucket Bucket, string? MappedKey) RouteKey(string key, bool vaeOriginalNaming)
    {
        if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8")
            return (LtxBucket.Drop, null);

        if (key.StartsWith(VaePrefix, StringComparison.Ordinal))
            return MapVae(key[VaePrefix.Length..], vaeOriginalNaming);
        if (key.StartsWith(TransformerPrefix, StringComparison.Ordinal))
            return MapTransformer(key[TransformerPrefix.Length..]);

        // Bare keys (diffusers folder shards): VAE keys are recognizable by their module roots.
        if (key.StartsWith("decoder.", StringComparison.Ordinal) || key.StartsWith("encoder.", StringComparison.Ordinal)
            || key.StartsWith("latents_", StringComparison.Ordinal) || key.StartsWith("per_channel_statistics.", StringComparison.Ordinal)
            || key.StartsWith("timestep_scale_multiplier", StringComparison.Ordinal))
            return MapVae(key, vaeOriginalNaming);
        return MapTransformer(key);
    }

    private static (LtxBucket, string?) MapTransformer(string key)
    {
        if (key.Contains("vae"))
            return (LtxBucket.Drop, null);
        foreach ((string from, string to) in _transformerRenames)
            key = key.Replace(from, to, StringComparison.Ordinal);
        return (LtxBucket.Transformer, key);
    }

    private static (LtxBucket, string?) MapVae(string key, bool originalNaming)
    {
        if (originalNaming)
            foreach ((string from, string to) in _vaeRenames)
                key = key.Replace(from, to, StringComparison.Ordinal);
        // Stats not renamed above (channel index, mean-of-stds) are metadata the decoder never reads.
        if (key.Contains("per_channel_statistics"))
            return (LtxBucket.Drop, null);
        return (LtxBucket.Vae, key);
    }

    /// <summary>Converts a flat weight dictionary (single file or merged shards) into transformer + VAE buckets.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);
        bool vaeOriginal = IsOriginalVaeNaming(allWeights.Keys);

        Dictionary<string, Tensor> transformer = new(allWeights.Count);
        Dictionary<string, Tensor> vae = new(512);
        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            (LtxBucket bucket, string? mapped) = RouteKey(kvp.Key, vaeOriginal);
            switch (bucket)
            {
                case LtxBucket.Transformer: transformer[mapped!] = kvp.Value; break;
                case LtxBucket.Vae: vae[mapped!] = kvp.Value; break;
                case LtxBucket.Drop: break;
            }
        }
        return new ConvertedWeights { Transformer = transformer, Vae = vae };
    }

    /// <summary>Loads a single safetensors file (bundled DiT + VAE) and converts. The caller owns the loader and
    /// disposes it once the weights are no longer referenced.</summary>
    public static (ConvertedWeights Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"LTX-Video checkpoint not found: {checkpointPath}");
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>Loads a diffusers folder layout: merges the safetensors shards of <paramref name="componentDir"/>
    /// (a <c>transformer/</c> or <c>vae/</c> subdirectory) and converts.</summary>
    public static (ConvertedWeights Weights, List<SafeTensorsLoader> Loaders) LoadDiffusersFolder(string componentDir)
    {
        if (!Directory.Exists(componentDir))
            throw new DirectoryNotFoundException($"LTX-Video component dir not found: {componentDir}");
        string[] shards = Directory.GetFiles(componentDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No safetensors shards found in: {componentDir}");
        Array.Sort(shards, StringComparer.Ordinal);

        List<SafeTensorsLoader> loaders = new(shards.Length);
        Dictionary<string, Tensor> merged = new(2048);
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
            return (Convert(merged), loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }
}
