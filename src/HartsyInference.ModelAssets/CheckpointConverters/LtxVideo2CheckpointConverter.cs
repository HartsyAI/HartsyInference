using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Routes an LTX-2.x (Lightricks, 22B audiovisual; 2.3 and 2.5) checkpoint into the per-component weight dictionaries the model code consumes: the DiT (<c>LtxVideo2Transformer</c>), the per-modality text connectors (<c>LtxVideo2TextConnectors</c>), the video VAEs (<c>LtxVideo2VaeDecoder</c> conv / <c>LtxVideo25DiffusionDecoder</c>), the audio VAE (<c>LtxAudioVaeDecoder</c>), the vocoder (<c>LtxAudioVocoder</c>), and — when bundled — the text tower (Gemma-3-12B on 2.3, Gemma-4-12B on 2.5).
///
/// <para>The single-file Lightricks checkpoint (<c>ltx-2.3-22b-dev.safetensors</c>) carries the DiT and connectors under <c>model.diffusion_model.*</c>, the video VAE under <c>vae.*</c>, the audio VAE under <c>audio_vae.*</c>, the vocoder under <c>vocoder.*</c>, and (optionally) the text encoder under <c>text_encoder.*</c>. Routing strips the component prefix where the consumer expects bare keys (DiT, both VAEs, text encoder) and keeps it where the consumer looks the key up with its prefix intact (the connectors read <c>model.diffusion_model.*_embeddings_connector.*</c> and the vocoder reads <c>vocoder.vocoder.*</c> / <c>vocoder.bwe_generator.*</c> / <c>vocoder.mel_stft.*</c>).</para>
///
/// <para><b>Status:</b> the rename table is verified against the real checkpoint header (see the table comment below), and the conv-VAE + audio paths have real-weight parity vs ComfyUI; DiT end-to-end numeric parity is still tracked in PARITY_VERIFICATION.md. Already-diffusers-named inputs (folder shards) pass through the prefix routing unchanged.</para></summary>
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
        /// <summary>LTX-2.5 <c>NADiffusionDecoder</c> weights, kept apart because it shares module names (<c>decoder.conv_in</c>, <c>decoder.conv_out</c>) with the convolutional decoder it replaces.</summary>
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

    /// <summary>True when a key belongs to the text-connector subtree (the per-modality connectors + the aggregate feature projection), which the connector code reads with its prefix intact.</summary>
    private static bool IsConnectorKey(string key) =>
        key.Contains("embeddings_connector", StringComparison.Ordinal)
        || key.Contains("text_embedding_projection", StringComparison.Ordinal);

    /// <summary>Whether a bare key belongs to a standalone Gemma 4 text tower (LTX-2.5's <c>gemma4-12b-with-proj</c> file) rather than to the DiT.</summary>
    /// <remarks>The multimodal heads (<c>vision_model</c>, <c>audio_projector</c>, <c>multi_modal_projector</c>) ride along in that file and are unused for LTX conditioning; they route here so they land somewhere the tower ignores instead of being mistaken for DiT weights.</remarks>
    private static bool IsGemma4TowerKey(string key) =>
        key.StartsWith("model.layers.", StringComparison.Ordinal)
        || key.StartsWith("model.embed_tokens.", StringComparison.Ordinal)
        || key.StartsWith("model.norm.", StringComparison.Ordinal)
        || key.StartsWith("multi_modal_projector.", StringComparison.Ordinal)
        || key.StartsWith("audio_projector.", StringComparison.Ordinal)
        || key.StartsWith("vision_model.", StringComparison.Ordinal)
        || key == "tokenizer_json";

    /// <summary>Key whose presence identifies an LTX-2.5 diffusion video VAE — the noised-pixel input projection, which the convolutional decoder has no counterpart for. Same signature ComfyUI selects on.</summary>
    public const string DiffusionDecoderSignatureKey = "decoder.conv_in_x_t.weight";

    /// <summary>Whether these keys belong to a diffusion video VAE rather than a convolutional one. This has to be a whole-file question: the two decoders share <c>decoder.conv_in</c>/<c>decoder.conv_out</c>, so no single key answers it.</summary>
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

    /// <summary>Whether a raw checkpoint key belongs to the DiT, used to pick which file of a split bundle carries the architecture metadata.</summary>
    public static bool IsTransformerKey(string key) => RouteKey(key).Bucket == Ltx2Bucket.Transformer;

    /// <summary>Pure key routing (testable without files): returns the destination bucket and the mapped key.</summary>
    public static (Ltx2Bucket Bucket, string? MappedKey) RouteKey(string key)
    {
        if (key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || key == "scaled_fp8")
            return (Ltx2Bucket.Drop, null);
        // Packagers embed side files (chat template, generation/processor/tokenizer config) as U8 tensors under
        // this prefix. They are not weights; without dropping them they fall through to the DiT mapper.
        if (key.StartsWith("hf_asset__", StringComparison.Ordinal))
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
        // LTX-2.5 ships its Gemma 4 tower as its own file with BARE `model.layers.*` keys — no `text_encoder.`
        // prefix — so without this they fall through to MapTransformer below and are loaded as DiT weights. The
        // DiT's own keys carry `model.diffusion_model.`, a different prefix, so it is unaffected. Checked after
        // IsConnectorKey because `text_embedding_projection.*` lives in the same file but belongs to the connectors.
        if (IsGemma4TowerKey(key))
            return (Ltx2Bucket.TextEncoder, key);
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

    /// <summary>Maps an original flat <c>up_blocks.{i}</c> token to the diffusers grouping by index parity: index 0 is the mid block; odd i is the (i−1)/2-th up-stage's upsampler; even i&gt;0 is the (i/2−1)-th up-stage's resnets. Handles any up-block count (19B has 7, 22B has 9). Keys without an <c>up_blocks.</c> token pass through.</summary>
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
    /// <param name="residentNvfp4">Keep an nvfp4 build packed instead of unpacking it to BF16 at load. Only a caller whose backend can consume a packed weight may ask for it — the official 18.72 GB LTX-2.5 distilled nvfp4 DiT unpacks to 42 GB, so this is the difference between fitting a 24 GB card and not.</param>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights, bool residentNvfp4 = false)
    {
        // Folds every quantized build's companions onto the weight before routing: the LTX-2.5 comfy-int8-convrot DiT
        // keeps its I8 bytes with the row scale on Tensor.QuantInfo, fp8 builds get Fp8ScaleFactor.
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights, residentNvfp4: residentNvfp4);

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

    /// <summary>Loads a single bundled safetensors file and routes it. The caller owns the loader and disposes it once the weights are no longer referenced.</summary>
    public static (ConvertedWeights Weights, SafeTensorsLoader Loader) LoadAndConvert(string checkpointPath)
    {
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"LTX-2 checkpoint not found: {checkpointPath}");

        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        ConvertedWeights converted = Convert(loader.GetAllTensors());
        return (converted, loader);
    }
}
