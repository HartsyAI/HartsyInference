using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Routes a MiniMax-H3 ("Hailuo 03") checkpoint into the per-component weight dictionaries the model code consumes: the DiT (<c>MiniMaxH3Transformer</c>), the video VAE, the audio VAE, and the Qwen3-VL-32B conditioning tower.
///
/// <para>The official release is a diffusers folder (<c>FL2VA/{transformer,video_vae,audio_vae,text_encoder}/</c>) in which every file is already flat and unprefixed, so DiT keys route through unchanged — verified key-for-key against the 535-tensor header of <c>minimax_h3_fl2va_bf16.safetensors</c> and the 902-tensor header of <c>qwen3vl_32b_minimax_h3_bf16.safetensors</c>. The dotted component prefixes (<c>model.diffusion_model.</c>, <c>vae.</c>, <c>audio_vae.</c>, <c>text_encoders.qwen3vl_32b.transformer.</c>) are supported defensively for a ComfyUI-style single-file bundle and are <b>not</b> verified against a real bundled H3 checkpoint.</para>
///
/// <para>Both VAE files are flat and share bare <c>encoder.</c>/<c>decoder.</c> roots, so an unprefixed VAE key is not distinguishable from the other VAE's — feed VAE component files to their own loaders; only the prefixed bundle form routes them here.</para>
///
/// <para>The DiT ships BF16 with 13 F32 tensors (both patch projections, both output heads, the time embedder, <c>rope.inv_freq</c>); <c>MiniMaxH3Transformer</c> consumes F32 throughout, so BF16/F16 tensors are materialized as F32 when <c>castToF32</c> is set. Cast tensors are caller-owned; uncast ones borrow the loader's mmap and die with it.</para>
///
/// <para>The pruned release swaps the time embedder for an <c>adaln_t_table</c> curve basis (supported — the config detects it and the transformer implements the lerp) and quantizes the four block linears to int8 with a block-diagonal input rotation (<c>convrot</c>). Those weights stay packed: their <c>.weight_scale</c> and <c>.comfy_quant</c> companions are folded onto <see cref="Tensor.QuantInfo"/> by <see cref="CheckpointConvertUtils.AttachInt8QuantInfo"/> and the backend consumes the int8 bytes directly.</para></summary>
public sealed class MiniMaxH3CheckpointConverter
{
    private const string DiffusionPrefix = "model.diffusion_model.";
    private const string BareDiffusionPrefix = "diffusion_model.";
    private const string AudioVaePrefix = "audio_vae.";
    private const string VaePrefix = "vae.";
    private const string VideoVaePrefix = "video_vae.";
    private const string TextEncodersPrefix = "text_encoders.qwen3vl_32b.transformer.";
    private const string TextEncoderPrefix = "text_encoder.";

    /// <summary>Destination bucket for a checkpoint key.</summary>
    public enum MiniMaxH3Bucket
    {
        /// <summary>DiT weights for <c>MiniMaxH3Transformer.LoadWeights</c> (flat, prefix stripped).</summary>
        Transformer,
        /// <summary>Video VAE weights (3D causal CNN encoder + ViT3D decoder).</summary>
        VideoVae,
        /// <summary>Audio VAE weights (DAC-lineage encoder + BigVGAN decoder).</summary>
        AudioVae,
        /// <summary>Qwen3-VL-32B conditioning tower weights (<c>model.*</c> + <c>visual.*</c>).</summary>
        TextEncoder,
        /// <summary>Unused (fp8 scale markers). Quantization companions never reach routing — <see cref="Convert"/> folds them onto their weight and drops them first.</summary>
        Drop,
    }

    /// <summary>Per-component weight buckets. Each maps the keys its consumer's <c>LoadWeights</c> looks up.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Transformer { get; init; }
        public required Dictionary<string, Tensor> VideoVae { get; init; }
        public required Dictionary<string, Tensor> AudioVae { get; init; }
        public required Dictionary<string, Tensor> TextEncoder { get; init; }
    }

    /// <summary>True when a bare (unprefixed) key belongs to the Qwen3-VL tower rather than the DiT.</summary>
    private static bool IsBareTextEncoderKey(string key) =>
        key.StartsWith("visual.", StringComparison.Ordinal) || key.StartsWith("model.layers.", StringComparison.Ordinal)
        || key.StartsWith("model.embed_tokens", StringComparison.Ordinal)
        || key.StartsWith("model.norm.", StringComparison.Ordinal)
        || key.StartsWith("lm_head.", StringComparison.Ordinal);

    /// <summary>Pure key routing (testable without files): returns the destination bucket and the mapped key.</summary>
    public static (MiniMaxH3Bucket Bucket, string? MappedKey) RouteKey(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8")
            return (MiniMaxH3Bucket.Drop, null);

        // Dotted-prefix order is load-bearing: "audio_vae." before "vae.", and the diffusion prefix before the bare
        // "model.*" text-encoder test, or "audio_patch_proj"/"model.diffusion_model" would fall into the wrong bucket.
        if (key.StartsWith(DiffusionPrefix, StringComparison.Ordinal))
            return (MiniMaxH3Bucket.Transformer, key[DiffusionPrefix.Length..]);
        if (key.StartsWith(BareDiffusionPrefix, StringComparison.Ordinal))
            return (MiniMaxH3Bucket.Transformer, key[BareDiffusionPrefix.Length..]);
        if (key.StartsWith(AudioVaePrefix, StringComparison.Ordinal))
            return (MiniMaxH3Bucket.AudioVae, key[AudioVaePrefix.Length..]);
        if (key.StartsWith(VideoVaePrefix, StringComparison.Ordinal))
            return (MiniMaxH3Bucket.VideoVae, key[VideoVaePrefix.Length..]);
        if (key.StartsWith(VaePrefix, StringComparison.Ordinal))
            return (MiniMaxH3Bucket.VideoVae, key[VaePrefix.Length..]);
        if (key.StartsWith(TextEncodersPrefix, StringComparison.Ordinal))
            return (MiniMaxH3Bucket.TextEncoder, key[TextEncodersPrefix.Length..]);
        if (key.StartsWith(TextEncoderPrefix, StringComparison.Ordinal))
            return (MiniMaxH3Bucket.TextEncoder, key[TextEncoderPrefix.Length..]);

        if (IsBareTextEncoderKey(key))
            return (MiniMaxH3Bucket.TextEncoder, key);
        return (MiniMaxH3Bucket.Transformer, key);
    }

    /// <summary>True when the flat dictionary carries the H3 DiT signature (both patch projections).</summary>
    public static bool IsMiniMaxH3(IReadOnlyDictionary<string, Tensor> allWeights)
    {
        ArgumentNullException.ThrowIfNull(allWeights);
        foreach (string key in allWeights.Keys)
        {
            (MiniMaxH3Bucket bucket, string? mapped) = RouteKey(key);
            if (bucket == MiniMaxH3Bucket.Transformer && mapped == "video_patch_proj.weight")
                return true;
        }
        return false;
    }

    /// <summary>Routes a flat weight dictionary (single file or merged shards) into the per-component buckets, casting BF16/F16 to F32 when <paramref name="castToF32"/> (the DiT consumer is F32-only).</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights, bool castToF32 = true)
    {
        ArgumentNullException.ThrowIfNull(allWeights);
        // Folds BOTH quantized builds' companions before routing (int8_tensorwise row scales onto Tensor.QuantInfo,
        // fp8 per-tensor scalars onto Fp8ScaleFactor) — left in place those keys look like unknown weights.
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new Dictionary<string, Tensor>(allWeights.Count);
        Dictionary<string, Tensor> videoVae = new Dictionary<string, Tensor>(512);
        Dictionary<string, Tensor> audioVae = new Dictionary<string, Tensor>(256);
        Dictionary<string, Tensor> textEncoder = new Dictionary<string, Tensor>(1024);

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            (MiniMaxH3Bucket bucket, string? mapped) = RouteKey(kvp.Key);
            switch (bucket)
            {
                case MiniMaxH3Bucket.Transformer: transformer[mapped!] = MaybeCast(kvp.Value, castToF32); break;
                case MiniMaxH3Bucket.VideoVae: videoVae[mapped!] = kvp.Value; break;
                case MiniMaxH3Bucket.AudioVae: audioVae[mapped!] = kvp.Value; break;
                case MiniMaxH3Bucket.TextEncoder: textEncoder[mapped!] = kvp.Value; break;
                case MiniMaxH3Bucket.Drop: break;
            }
        }
        return new ConvertedWeights
        {
            Transformer = transformer,
            VideoVae = videoVae,
            AudioVae = audioVae,
            TextEncoder = textEncoder,
        };
    }

    /// <summary>F32 is required by the transformer; quantized and already-F32 tensors pass through untouched.</summary>
    private static Tensor MaybeCast(Tensor tensor, bool castToF32) =>
        CodecKeyUtils.MaybeCast(tensor, castToF32 && (tensor.DType == DType.BF16 || tensor.DType == DType.F16));

    /// <summary>Loads one safetensors file (a diffusers component file or a bundled checkpoint) and routes it. The caller owns the loader and disposes it once the weights are no longer referenced.</summary>
    public static (ConvertedWeights Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath,
        bool castToF32 = true)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"MiniMax-H3 checkpoint not found: {checkpointPath}");

        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(checkpointPath);
        try
        {
            return (Convert(loader.GetAllTensors(), castToF32), loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Loads and merges the <c>*.safetensors</c> shards of ONE component directory (e.g. <c>FL2VA/transformer</c> or a sharded <c>text_encoder</c>), then routes the merged dictionary. Deliberately non-recursive: both H3 VAEs use bare <c>encoder.</c>/<c>decoder.</c> roots, so merging the whole FL2VA tree would be unroutable.</summary>
    public static (ConvertedWeights Weights, List<SafeTensorsLoader> Loaders) LoadAndConvertShards(string dir,
        bool castToF32 = true)
    {
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException($"MiniMax-H3 checkpoint dir not found: {dir}");
        string[] shards = Directory.GetFiles(dir, "*.safetensors", SearchOption.TopDirectoryOnly);
        if (shards.Length == 0)
            throw new FileNotFoundException($"No safetensors shards found in: {dir}");
        Array.Sort(shards, StringComparer.Ordinal);

        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>(shards.Length);
        Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(4096);
        try
        {
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new SafeTensorsLoader();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                    merged[kvp.Key] = kvp.Value;
            }
            return (Convert(merged, castToF32), loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
            throw;
        }
    }
}
