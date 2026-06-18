using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.ModelHandler.CheckpointConverters;

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
    // sequentially per key. The connector subtrees and the q_norm/k_norm QK-norms are intentionally NOT renamed —
    // the connector and attention code read those names verbatim.
    private static readonly (string From, string To)[] _ditRenames =
    [
        ("patchify_proj", "proj_in"),
        ("audio_patchify_proj", "audio_proj_in"),
    ];

    /// <summary>Destination bucket for a checkpoint key.</summary>
    public enum Ltx2Bucket
    {
        /// <summary>DiT weights for <c>LtxVideo2Transformer.LoadWeights</c> (prefix stripped).</summary>
        Transformer,
        /// <summary>Text-connector weights for <c>LtxVideo2TextConnectors.LoadWeights</c> (prefix kept).</summary>
        Connectors,
        /// <summary>Video VAE decoder weights (<c>decoder.*</c> + <c>latents_mean</c>/<c>latents_std</c>).</summary>
        Vae,
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
        public required Dictionary<string, Tensor> AudioVae { get; init; }
        public required Dictionary<string, Tensor> Vocoder { get; init; }
        public required Dictionary<string, Tensor> TextEncoder { get; init; }
    }

    /// <summary>True when a key belongs to the text-connector subtree (the per-modality connectors + the aggregate
    /// feature projection), which the connector code reads with its prefix intact.</summary>
    private static bool IsConnectorKey(string key) =>
        key.Contains("embeddings_connector", StringComparison.Ordinal)
        || key.Contains("text_embedding_projection", StringComparison.Ordinal);

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
            || key.StartsWith("latents_", StringComparison.Ordinal))
            return MapVae(key);
        return MapTransformer(key);
    }

    private static (Ltx2Bucket, string?) MapTransformer(string key)
    {
        foreach ((string from, string to) in _ditRenames)
            key = key.Replace(from, to, StringComparison.Ordinal);
        return (Ltx2Bucket.Transformer, key);
    }

    private static (Ltx2Bucket, string?) MapVae(string key)
    {
        if (key.Contains("per_channel_statistics", StringComparison.Ordinal))
            return (Ltx2Bucket.Drop, null);
        return (Ltx2Bucket.Vae, key);
    }

    private static (Ltx2Bucket, string?) MapAudioVae(string key)
    {
        if (key.Contains("per_channel_statistics", StringComparison.Ordinal))
            return (Ltx2Bucket.Drop, null);
        return (Ltx2Bucket.AudioVae, key);
    }

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

        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            (Ltx2Bucket bucket, string? mapped) = RouteKey(kvp.Key);
            switch (bucket)
            {
                case Ltx2Bucket.Transformer: transformer[mapped!] = kvp.Value; break;
                case Ltx2Bucket.Connectors: connectors[mapped!] = kvp.Value; break;
                case Ltx2Bucket.Vae: vae[mapped!] = kvp.Value; break;
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
