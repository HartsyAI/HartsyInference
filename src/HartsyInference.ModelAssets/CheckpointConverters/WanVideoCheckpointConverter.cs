using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads Wan-Video DiT checkpoints (Wan2.2 TI2V-5B and family) into the diffusers state-dict names that <see cref="HartsyInference.Diffusion.Models.Denoisers.WanVideoTransformer"/> expects.
///
/// <para>Two real-world layouts are handled: (1) the <c>Wan-AI/*-Diffusers</c> repos, whose <c>transformer/</c> shards are already in diffusers naming (pass-through after prefix strip + FP8 scale folding); (2) original/ComfyUI repackaged single files (e.g. <c>wan2.2_ti2v_5B_fp16.safetensors</c>) in the original Wan naming (<c>blocks.{i}.self_attn.q</c>, <c>time_embedding.0</c>, <c>head.head</c>, …), which are renamed via the ordered rule table ported verbatim from diffusers <c>scripts/convert_wan_to_diffusers.py</c> — including the <c>norm2</c>/<c>norm3</c> swap (original <c>norm3</c> is the cross-attn norm = diffusers <c>norm2</c>).</para>
///
/// <para><b>VAE note:</b> the TI2V-5B VAE is the already-built <c>Wan22VaeDecoder</c> (original Wan naming) — load it with <see cref="LanceCheckpointConverter.LoadVae"/> (same <c>wan2.2_vae.safetensors</c> Lance uses). The diffusers <c>vae/</c> folder layout (<c>AutoencoderKLWan</c> naming) is NOT supported.</para>
///
/// <para><b>Wan2.2-Animate:</b> the animate-specific keys (<c>pose_patch_embedding.*</c>, <c>motion_encoder.enc.net_app.convs.{i}.*</c> / <c>enc.fc.{i}.*</c> / <c>dec.direction.weight</c>, <c>face_encoder.conv{1_local,2,3}.conv.*</c> / <c>out_proj.*</c> / <c>padding_tokens</c>, <c>face_adapter.fuser_blocks.{i}.*</c>, <c>ref_conv.*</c>) match NO rename rule and pass through unchanged — <c>WanAnimateTransformer.LoadWeights</c> expects them under their original names (pinned by <c>WanVideoCheckpointConverterTests.MapKey_AnimateKeys_PassThroughUnchanged</c>). The base i2v keys (<c>img_emb.proj.*</c>, <c>cross_attn.k_img</c>/<c>v_img</c>/<c>norm_k_img</c>) convert to the diffusers names as usual.</para>
///
/// <para><b>Wan-Animate-2:</b> the checkpoint IS a Wan2.1 I2V-14B one — same 40 blocks, same <c>img_emb</c>, none of the V1 pose/face surface — so it renames through the same rules and is recognised only by its <c>__metadata__</c> (<see cref="IsAnimate2Metadata"/>). Its int8-convrot quantization needs no arm of its own: the <c>.weight_scale</c> / <c>.comfy_quant</c> companions ride the shared <see cref="CheckpointConvertUtils.AttachInt8QuantInfo"/> pass.</para></summary>
public sealed class WanVideoCheckpointConverter
{
    private static readonly string[] _stripPrefixes = ["model.diffusion_model.", "diffusion_model.", "transformer.", "model."];

    // Ported verbatim (and in order — the rules are applied sequentially per key, like the python script's
    // key.replace chain; the norm2→placeholder→norm3 swap depends on it) from diffusers convert_wan_to_diffusers.py.
    private static readonly (string From, string To)[] _originalRenames =
    [
        ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
        ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
        ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
        ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
        ("time_projection.1", "condition_embedder.time_proj"),
        ("head.modulation", "scale_shift_table"),
        ("head.head", "proj_out"),
        ("modulation", "scale_shift_table"),
        ("ffn.0", "ffn.net.0.proj"),
        ("ffn.2", "ffn.net.2"),
        ("norm2", "norm__placeholder"),
        ("norm3", "norm2"),
        ("norm__placeholder", "norm3"),
        ("img_emb.proj.0", "condition_embedder.image_embedder.norm1"),
        ("img_emb.proj.1", "condition_embedder.image_embedder.ff.net.0.proj"),
        ("img_emb.proj.3", "condition_embedder.image_embedder.ff.net.2"),
        ("img_emb.proj.4", "condition_embedder.image_embedder.norm2"),
        ("img_emb.emb_pos", "condition_embedder.image_embedder.pos_embed"),
        ("self_attn.q", "attn1.to_q"),
        ("self_attn.k", "attn1.to_k"),
        ("self_attn.v", "attn1.to_v"),
        ("self_attn.o", "attn1.to_out.0"),
        ("self_attn.norm_q", "attn1.norm_q"),
        ("self_attn.norm_k", "attn1.norm_k"),
        ("cross_attn.q", "attn2.to_q"),
        ("cross_attn.k", "attn2.to_k"),
        ("cross_attn.v", "attn2.to_v"),
        ("cross_attn.o", "attn2.to_out.0"),
        ("cross_attn.norm_q", "attn2.norm_q"),
        ("cross_attn.norm_k", "attn2.norm_k"),
        ("attn2.to_k_img", "attn2.add_k_proj"),
        ("attn2.to_v_img", "attn2.add_v_proj"),
        ("attn2.norm_k_img", "attn2.norm_added_k"),
        // VACE control branch (diffusers convert script's VACE additions): the vace_blocks share the main block
        // sub-names (handled by the rules above); only the per-block control projections differ.
        ("before_proj", "proj_in"),
        ("after_proj", "proj_out"),
    ];

    /// <summary>Result bucket from a Wan DiT checkpoint. The umT5 text encoder + VAE ship separately.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>DiT weights in diffusers naming for <c>WanVideoTransformer.LoadWeights</c>.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }

        /// <summary>True when the file's <c>__metadata__</c> declared Wan-Animate-2 (<c>WanAnimate2Transformer</c>).</summary>
        public bool IsAnimate2 { get; init; }
    }

    /// <summary><c>__metadata__</c> entry whose JSON value carries the ComfyUI model type.</summary>
    private const string ConfigMetadataKey = "config";

    /// <summary>Reads <c>__metadata__["config"] = {"transformer": {"model_type": "animate2"}}</c>. This is the ONLY reliable Animate-2 discriminator: its key set is byte-for-byte a Wan2.1 I2V-14B one, so <see cref="HasAnimate2Structure"/> can validate the claim but can never make it.</summary>
    public static bool IsAnimate2Metadata(IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || !metadata.TryGetValue(ConfigMetadataKey, out string? config))
            return false;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(config);
            return doc.RootElement.TryGetProperty("transformer", out JsonElement transformer)
                && transformer.TryGetProperty("model_type", out JsonElement modelType)
                && modelType.ValueKind == JsonValueKind.String && modelType.GetString() == "animate2";
        }
        catch (JsonException) { return false; }
    }

    /// <summary>True when a (prefix-stripped, original-naming) key set is shaped like Animate-2: the i2v CLIP <c>img_emb</c> is present and none of the Animate-V1 conditioning surface is. Never a classifier on its own — a plain Wan2.1 I2V-14B checkpoint satisfies it too; it only rejects a file that claims to be Animate-2 and carries the V1 pose/face modules (which <c>WanAnimate2Transformer</c> would silently drop).</summary>
    public static bool HasAnimate2Structure(IEnumerable<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        bool imgEmb = false, blocks = false;
        foreach (string raw in keys)
        {
            string key = StripPrefix(raw);
            if (key.StartsWith("pose_patch_embedding.", StringComparison.Ordinal)
                || key.StartsWith("motion_encoder.", StringComparison.Ordinal)
                || key.StartsWith("face_encoder.", StringComparison.Ordinal)
                || key.StartsWith("face_adapter.", StringComparison.Ordinal))
            {
                return false;
            }
            imgEmb |= key.StartsWith("img_emb.proj.", StringComparison.Ordinal)
                || key.StartsWith("condition_embedder.image_embedder.", StringComparison.Ordinal);
            blocks |= key.StartsWith("blocks.0.", StringComparison.Ordinal);
        }
        return imgEmb && blocks;
    }

    /// <summary>True when the (prefix-stripped) key set uses the original Wan naming rather than diffusers naming.</summary>
    public static bool IsOriginalNaming(IEnumerable<string> keys)
    {
        foreach (string raw in keys)
        {
            string key = StripPrefix(raw);
            if (key.Contains("self_attn.") || key.Contains("cross_attn.")
                || key.StartsWith("time_embedding.", StringComparison.Ordinal)
                || key.StartsWith("head.head", StringComparison.Ordinal))
                return true;
            if (key.Contains("attn1.") || key.Contains("condition_embedder."))
                return false;
        }
        return false;
    }

    /// <summary>Pure key mapping (testable without files): strips the single-file prefix and, when <paramref name="fromOriginalNaming"/>, applies the ordered original→diffusers renames. Returns <c>null</c> for keys the transformer never consumes (FP8 scale metadata is folded separately in <see cref="Convert"/>).</summary>
    public static string? MapKey(string key, bool fromOriginalNaming)
    {
        if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8")
            return null;
        string mapped = StripPrefix(key);
        if (fromOriginalNaming)
            foreach ((string from, string to) in _originalRenames)
                mapped = mapped.Replace(from, to, StringComparison.Ordinal);
        return mapped;
    }

    /// <summary>Converts a flat weight dictionary (single file or merged shards) to the diffusers-named transformer bucket. <paramref name="metadata"/> is the file's <c>__metadata__</c>, read only for the Animate-2 declaration.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights, IReadOnlyDictionary<string, string>? metadata = null)
    {
        bool animate2 = IsAnimate2Metadata(metadata);
        if (animate2 && !HasAnimate2Structure(allWeights.Keys))
        {
            throw new NotSupportedException(
                "Checkpoint declares model_type 'animate2' but carries the Animate-V1 pose/face conditioning modules "
                + "(or no img_emb). Wan-Animate-2 has neither pose_patch_embedding nor motion_encoder/face_encoder/face_adapter.");
        }
        // int8_tensorwise/fp8 companions move onto QuantInfo here, BEFORE the rename pass, so `.weight_scale` and
        // `.comfy_quant` never reach MapKey.
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);
        bool original = IsOriginalNaming(allWeights.Keys);

        Dictionary<string, Tensor> transformer = new(allWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            string? mapped = MapKey(kvp.Key, original);
            if (mapped is not null) transformer[mapped] = kvp.Value;
        }
        return new ConvertedWeights { Transformer = transformer, IsAnimate2 = animate2 };
    }

    /// <summary>Loads a single safetensors file and converts. The caller owns the loader and disposes it once the weights are no longer referenced.</summary>
    public static (ConvertedWeights Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"Wan-Video checkpoint not found: {checkpointPath}");
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors(), loader.Metadata);
        return (converted, loader);
    }

    /// <summary>Loads a diffusers <c>transformer/</c> folder (sharded safetensors, already diffusers-named) and converts.</summary>
    public static (ConvertedWeights Weights, List<SafeTensorsLoader> Loaders) LoadDiffusersFolder(string transformerDir)
    {
        if (!Directory.Exists(transformerDir))
            throw new DirectoryNotFoundException($"Wan-Video transformer dir not found: {transformerDir}");
        string[] shards = Directory.GetFiles(transformerDir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No safetensors shards found in: {transformerDir}");
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

    private static string StripPrefix(string key)
    {
        foreach (string prefix in _stripPrefixes)
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                return key[prefix.Length..];
        return key;
    }
}
