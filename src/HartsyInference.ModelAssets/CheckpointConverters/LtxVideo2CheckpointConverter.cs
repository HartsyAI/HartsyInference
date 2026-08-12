using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Routes an LTX-2.3 (Lightricks, 22B audiovisual) checkpoint into the per-component weight dictionaries the
/// model code consumes: the DiT (<c>LtxVideo2Transformer</c>), the per-modality text connectors
/// (<c>LtxVideo2TextConnectors</c>), the video VAE (<c>LtxVideo2VaeDecoder</c>), the audio VAE
/// (<c>LtxAudioVaeDecoder</c>), the vocoder (<c>LtxAudioVocoder</c>), and — when bundled — the Gemma-3-12B text tower.
///
/// <para>The single-file Lightricks checkpoint (<c>ltx-2.3-22b-dev.safetensors</c>) carries the DiT and connectors
/// under <c>model.diffusion_model.*</c>, the video VAE under <c>vae.*</c>, the audio VAE under <c>audio_vae.*</c>, the
/// vocoder under <c>vocoder.*</c>, and (optionally) the text encoder under <c>text_encoder.*</c>. Routing strips the
/// component prefix where the consumer expects bare keys (DiT, both VAEs, text encoder) and keeps it where the
/// consumer looks the key up with its prefix intact (the connectors read <c>model.diffusion_model.*_embeddings_connector.*</c>
/// and the vocoder reads <c>vocoder.vocoder.*</c> / <c>vocoder.bwe_generator.*</c> / <c>vocoder.mel_stft.*</c>).</para>
///
/// <para><b>Status:</b> structural — routing matches the model code's <c>LoadWeights</c> contracts as written; the
/// exact original-naming rename table for the DiT sub-modules is validation-pending against the real checkpoint
/// header (the whole LTX-2 path is numerics-unverified). Already-diffusers-named inputs (folder shards) pass through
/// the prefix routing unchanged.</para></summary>
public sealed class LtxVideo2CheckpointConverter
{
    private const string DiffusionPrefix = "model.diffusion_model.";
    private const string VaePrefix = "vae.";
    private const string AudioVaePrefix = "audio_vae.";
    private const string VocoderPrefix = "vocoder.";
    private const string TextEncoderPrefix = "text_encoder.";

    // DiT sub-module renames (original Lightricks → the diffusers-style names LtxVideo2Transformer reads). Applied
    // sequentially per key via substring replace, so ORDER MATTERS: the most specific patterns come first so a
    // shorter one (e.g. "adaln_single") can't corrupt a longer key (e.g. "prompt_adaln_single") that contains it.
    // The connector subtrees and the q_norm/k_norm QK-norms are intentionally NOT renamed — the connector and
    // attention code read those names verbatim. Verified against the checkpoint header's `model.diffusion_model.*`
    // top-level module names (adaln_single → time_embed, etc.).
    private static readonly (string From, string To)[] _ditRenames =
    [
        ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
        ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
        ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
        ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
        ("audio_prompt_adaln_single", "audio_prompt_adaln"),
        ("prompt_adaln_single", "prompt_adaln"),
        ("audio_adaln_single", "audio_time_embed"),
        ("adaln_single", "time_embed"),
        ("audio_patchify_proj", "audio_proj_in"),
        ("patchify_proj", "proj_in"),
    ];

    // Video-VAE regroup: the single-file LTX-2 VAE is in ORIGINAL Lightricks naming — a FLAT decoder.up_blocks.0..N
    // list (verified from the 19B [N=6] and 22B [N=8] checkpoint headers): index 0 = the mid block's res blocks, then
    // odd = an upsample conv, even>0 = an up-stage's res blocks. The count varies per model (3 up-stages for the 19B, 4
    // for the 22B), so the regroup is INDEX-PARITY based (see RegroupUpBlock) rather than a fixed table. The
    // LtxVideo2VaeDecoder consumes the diffusers grouping (mid_block / up_blocks.{k}.upsamplers.0 / resnets); the
    // encoder (down_blocks.*) is carried through unrenamed (the decoder doesn't read it).

    /// <summary>Destination bucket for a checkpoint key.</summary>
    public enum Ltx2Bucket
    {
        /// <summary>DiT weights for <c>LtxVideo2Transformer.LoadWeights</c> (prefix stripped).</summary>
        Transformer,
        /// <summary>Text-connector weights for <c>LtxVideo2TextConnectors.LoadWeights</c> (prefix kept).</summary>
        Connectors,
        /// <summary>Video VAE decoder weights (<c>decoder.*</c> + <c>latents_mean</c>/<c>latents_std</c>).</summary>
        Vae,
        /// <summary>LTX-2.5 <c>NADiffusionDecoder</c> weights, kept apart because it shares module names
        /// (<c>decoder.conv_in</c>, <c>decoder.conv_out</c>) with the convolutional decoder it replaces.</summary>
        VaeDiffusionDecoder,
        /// <summary>Audio VAE decoder weights.</summary>
        AudioVae,
        /// <summary>Vocoder weights (prefix kept — consumer reads <c>vocoder.*</c>).</summary>
        Vocoder,
        /// <summary>Gemma-3-12B text tower weights (prefix stripped), if bundled.</summary>
        TextEncoder,
        /// <summary>Unused (fp8 scale companions, leftover stats).</summary>
        Drop,
    }

    /// <summary>Per-component weight buckets. Each maps the keys its consumer's <c>LoadWeights</c> looks up.</summary>
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Transformer { get; init; }
        public required Dictionary<string, Tensor> Connectors { get; init; }
        public required Dictionary<string, Tensor> Vae { get; init; }
        /// <summary>Populated only for an LTX-2.5 diffusion video VAE; empty for the convolutional decoder.</summary>
        public Dictionary<string, Tensor> VaeDiffusionDecoder { get; init; } = [];
        public required Dictionary<string, Tensor> AudioVae { get; init; }
        public required Dictionary<string, Tensor> Vocoder { get; init; }
        public required Dictionary<string, Tensor> TextEncoder { get; init; }
    }

    /// <summary>True when a key belongs to the text-connector subtree (the per-modality connectors + the aggregate
    /// feature projection), which the connector code reads with its prefix intact.</summary>
    private static bool IsConnectorKey(string key) =>
        key.Contains("embeddings_connector", StringComparison.Ordinal)
        || key.Contains("text_embedding_projection", StringComparison.Ordinal);

    /// <summary>Key whose presence identifies an LTX-2.5 diffusion video VAE — the noised-pixel input projection,
    /// which the convolutional decoder has no counterpart for. Same signature ComfyUI selects on.</summary>
    public const string DiffusionDecoderSignatureKey = "decoder.conv_in_x_t.weight";

    /// <summary>Whether these keys belong to a diffusion video VAE rather than a convolutional one. This has to be a
    /// whole-file question: the two decoders share <c>decoder.conv_in</c>/<c>decoder.conv_out</c>, so no single key
    /// answers it.</summary>
    public static bool IsDiffusionVideoVae(IEnumerable<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        foreach (string key in keys)
        {
            if (key == DiffusionDecoderSignatureKey
                || key.EndsWith($".{DiffusionDecoderSignatureKey}", StringComparison.Ordinal)
                || key == $"{VaePrefix}{DiffusionDecoderSignatureKey}")
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Whether a raw checkpoint key belongs to the DiT, used to pick which file of a split bundle carries
    /// the architecture metadata.</summary>
    public static bool IsTransformerKey(string key) => RouteKey(key).Bucket == Ltx2Bucket.Transformer;

    /// <summary>Pure key routing (testable without files): returns the destination bucket and the mapped key.</summary>
    public static (Ltx2Bucket Bucket, string? MappedKey) RouteKey(string key)
    {
        if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8")
            return (Ltx2Bucket.Drop, null);

        // Vocoder/audio VAE prefixes are matched before the bare "vae." so "audio_vae." never falls through to it.
        if (key.StartsWith(VocoderPrefix, StringComparison.Ordinal))
            return (Ltx2Bucket.Vocoder, key);                          // consumer reads the "vocoder." prefix
        if (key.StartsWith(AudioVaePrefix, StringComparison.Ordinal))
            return MapAudioVae(key[AudioVaePrefix.Length..]);
        if (key.StartsWith(VaePrefix, StringComparison.Ordinal))
            return MapVae(key[VaePrefix.Length..]);
        if (key.StartsWith(TextEncoderPrefix, StringComparison.Ordinal))
            return (Ltx2Bucket.TextEncoder, key[TextEncoderPrefix.Length..]);

        if (key.StartsWith(DiffusionPrefix, StringComparison.Ordinal))
        {
            // Connector keys keep the full prefix; everything else is DiT and gets stripped + renamed.
            if (IsConnectorKey(key))
                return (Ltx2Bucket.Connectors, key);
            return MapTransformer(key[DiffusionPrefix.Length..]);
        }

        // Bare keys (diffusers folder shards) routed by module root.
        if (IsConnectorKey(key))
            return (Ltx2Bucket.Connectors, key);
        if (key.StartsWith("decoder.", StringComparison.Ordinal) || key.StartsWith("encoder.", StringComparison.Ordinal)
            || key.StartsWith("latents_", StringComparison.Ordinal)
            || key.StartsWith("per_channel_statistics", StringComparison.Ordinal))
            return MapVae(key);
        return MapTransformer(key);
    }

    private static (Ltx2Bucket, string?) MapTransformer(string key)
    {
        foreach ((string from, string to) in _ditRenames)
            key = key.Replace(from, to, StringComparison.Ordinal);
        return (Ltx2Bucket.Transformer, key);
    }

    // Routes a (vae.-stripped) video VAE key. Single-file checkpoints are in original Lightricks naming and get
    // regrouped to the diffusers names the decoder reads; already-diffusers keys (folder shards: mid_block / resnets)
    // pass through. The per-channel stats become latents_mean/std (the loader reads them for un-normalization); the
    // other stats entries are dropped.
    private static (Ltx2Bucket, string?) MapVae(string key)
    {
        // Already-diffusers keys (folder shards) carry the grouping words and must NOT be regrouped.
        bool diffusersAlready = key.Contains("resnets", StringComparison.Ordinal)
            || key.Contains("upsamplers", StringComparison.Ordinal)
            || key.Contains("mid_block", StringComparison.Ordinal);
        bool original = !diffusersAlready
            && (key.Contains("up_blocks.", StringComparison.Ordinal)
                || key.Contains("res_blocks", StringComparison.Ordinal)
                || key.StartsWith("per_channel_statistics", StringComparison.Ordinal));
        if (original)
        {
            // Drop the stats we don't consume (channel index, mean-of-stds, …) — keep only the two renamed below.
            if (key.StartsWith("per_channel_statistics", StringComparison.Ordinal))
            {
                if (key.EndsWith("mean-of-means", StringComparison.Ordinal)) return (Ltx2Bucket.Vae, "latents_mean");
                if (key.EndsWith("std-of-means", StringComparison.Ordinal)) return (Ltx2Bucket.Vae, "latents_std");
                return (Ltx2Bucket.Drop, null);
            }
            key = RegroupUpBlock(key);
            key = key.Replace("res_blocks", "resnets", StringComparison.Ordinal);
        }
        return (Ltx2Bucket.Vae, key);
    }

    /// <summary>Maps an original flat <c>up_blocks.{i}</c> token to the diffusers grouping by index parity: index 0 is
    /// the mid block; odd i is the (i−1)/2-th up-stage's upsampler; even i&gt;0 is the (i/2−1)-th up-stage's resnets.
    /// Handles any up-block count (19B has 7, 22B has 9). Keys without an <c>up_blocks.</c> token pass through.</summary>
    private static string RegroupUpBlock(string key)
    {
        const string tok = "up_blocks.";
        int at = key.IndexOf(tok, StringComparison.Ordinal);
        if (at < 0) return key;
        int numStart = at + tok.Length, numEnd = numStart;
        while (numEnd < key.Length && char.IsDigit(key[numEnd])) numEnd++;
        if (numEnd == numStart) return key;
        int i = int.Parse(key.AsSpan(numStart, numEnd - numStart));
        string mapped = i == 0 ? "mid_block"
            : (i % 2 == 1) ? $"up_blocks.{(i - 1) / 2}.upsamplers.0"
            : $"up_blocks.{i / 2 - 1}";
        return string.Concat(key.AsSpan(0, at), mapped, key.AsSpan(numEnd));
    }

    private static (Ltx2Bucket, string?) MapAudioVae(string key) => (Ltx2Bucket.AudioVae, key);

    /// <summary>Routes a flat weight dictionary (single file or merged shards) into the per-component buckets.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(allWeights.Count);
        Dictionary<string, Tensor> connectors = new(256);
        Dictionary<string, Tensor> vae = new(512);
        Dictionary<string, Tensor> audioVae = new(256);
        Dictionary<string, Tensor> vocoder = new(256);
        Dictionary<string, Tensor> textEncoder = new(1024);
        Dictionary<string, Tensor> diffusionDecoder = [];

        // The latent statistics stay in the VAE bucket for either decoder — both un-normalize the latent the same way.
        bool diffusionVae = IsDiffusionVideoVae(allWeights.Keys);

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            (Ltx2Bucket bucket, string? mapped) = RouteKey(kvp.Key);
            if (diffusionVae && bucket == Ltx2Bucket.Vae && mapped is not null
                && mapped.StartsWith("decoder.", StringComparison.Ordinal))
            {
                bucket = Ltx2Bucket.VaeDiffusionDecoder;
            }
            switch (bucket)
            {
                case Ltx2Bucket.Transformer: transformer[mapped!] = kvp.Value; break;
                case Ltx2Bucket.Connectors: connectors[mapped!] = kvp.Value; break;
                case Ltx2Bucket.Vae: vae[mapped!] = kvp.Value; break;
                case Ltx2Bucket.VaeDiffusionDecoder: diffusionDecoder[mapped!] = kvp.Value; break;
                case Ltx2Bucket.AudioVae: audioVae[mapped!] = kvp.Value; break;
                case Ltx2Bucket.Vocoder: vocoder[mapped!] = kvp.Value; break;
                case Ltx2Bucket.TextEncoder: textEncoder[mapped!] = kvp.Value; break;
                case Ltx2Bucket.Drop: break;
            }
        }
        return new ConvertedWeights
        {
            Transformer = transformer,
            Connectors = connectors,
            Vae = vae,
            VaeDiffusionDecoder = diffusionDecoder,
            AudioVae = audioVae,
            Vocoder = vocoder,
            TextEncoder = textEncoder,
        };
    }

    /// <summary>Loads a single bundled safetensors file and routes it. The caller owns the loader and disposes it
    /// once the weights are no longer referenced.</summary>
    public static (ConvertedWeights Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"LTX-2 checkpoint not found: {checkpointPath}");

        // fp8 (opt-in, HARTSY_LTX2_FP8=1): the LTX-2.3 DiT ships in BF16 (~35 GB, streams on a 24 GB card). Build a
        // persistent fp8 REPACK once (quantize the DiT block Linears to fp8_scaled → ~18 GB, fully resident, ~6×
        // faster steps + graph-eligible), save it next to the checkpoint, and reuse it on every future run — no
        // re-quantization. The repack is a drop-in fp8 checkpoint that can also be shared/uploaded as an official
        // download. NOT default: the user opts into fp8 for the model; without it the original BF16 is used unchanged.
        if (Environment.GetEnvironmentVariable("HARTSY_LTX2_FP8") == "1")
        {
            string repackPath = checkpointPath[..^".safetensors".Length] + ".dit-fp8.safetensors";
            if (File.Exists(repackPath))
            {
                HartsyInference.Core.Logging.Logs.Info($"[LTX-2 fp8] using cached fp8 repack: {Path.GetFileName(repackPath)}");
                checkpointPath = repackPath;   // load the cached repack through the normal path below
            }
            else
            {
                // Building the repack needs room for both the original and the (smaller) fp8 file during the write.
                // If disk is short, DON'T fill it — quantize in memory this run instead (same speed, re-done next
                // cold load; free disk to cache it). Guards against the disk-full crash.
                long needBytes = new FileInfo(checkpointPath).Length + (2L << 30);
                long freeBytes = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(repackPath))!).AvailableFreeSpace;
                SafeTensorsLoader loader0 = new();
                loader0.Load(checkpointPath);
                if (freeBytes >= needBytes)
                {
                    HartsyInference.Core.Logging.Logs.Info($"[LTX-2 fp8] building fp8 repack (one-time): {Path.GetFileName(repackPath)}");
                    Dictionary<string, Tensor> all = new(loader0.GetAllTensors());
                    CheckpointConvertUtils.QuantizeDitBlocksToFp8(all, "diffusion_model.transformer_blocks.");
                    SafeTensorsWriter.Save(repackPath, all);
                    HartsyInference.Core.Logging.Logs.Info($"[LTX-2 fp8] fp8 repack saved ({new FileInfo(repackPath).Length >> 30} GB); future loads reuse it.");
                    loader0.Dispose();
                    checkpointPath = repackPath;   // reload the repack cleanly below
                }
                else
                {
                    HartsyInference.Core.Logging.Logs.Warning(
                        $"[LTX-2 fp8] not enough disk to cache the fp8 repack ({freeBytes >> 30} GB free, need ~{needBytes >> 30} GB) — " +
                        $"quantizing in memory this run (free disk to save a reusable repack). ");
                    ConvertedWeights conv = Convert(loader0.GetAllTensors());
                    CheckpointConvertUtils.QuantizeDitBlocksToFp8(conv.Transformer, "transformer_blocks.");
                    return (conv, loader0);
                }
            }
        }

        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }

    /// <summary>Loads and merges every <c>*.safetensors</c> shard in a directory (a bundled multi-shard checkpoint or
    /// a diffusers repo root), then routes the merged dictionary.</summary>
    public static (ConvertedWeights Weights, List<SafeTensorsLoader> Loaders) LoadAndConvertShards(string dir)
    {
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException($"LTX-2 checkpoint dir not found: {dir}");
        string[] shards = Directory.GetFiles(dir, "*.safetensors", SearchOption.AllDirectories);
        if (shards.Length == 0)
            throw new FileNotFoundException($"No safetensors shards found in: {dir}");
        Array.Sort(shards, StringComparer.Ordinal);

        List<SafeTensorsLoader> loaders = new(shards.Length);
        Dictionary<string, Tensor> merged = new(4096);
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
