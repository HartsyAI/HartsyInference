using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Resolves an <see cref="LtxVideo2Config"/> from a checkpoint's safetensors metadata and tensor keys,
/// so an LTX-2 point release is not pinned to whichever preset the recipe happened to hardcode.</summary>
/// <remarks>Every shipped LTX-2 checkpoint carries its architecture under <c>__metadata__["config"]["transformer"]</c>,
/// but repacks routinely strip metadata while never dropping a weight, so tensor keys are the fallback — and, for
/// the keyframe embedding, the authority. That ordering mirrors ComfyUI, whose key probe overwrites the value it
/// just merged from metadata (<c>comfy/model_detection.py</c>).</remarks>
public static class LtxVideo2VariantDetector
{
    /// <summary>Key that marks an LTX-2.5-generation checkpoint, in post-conversion (transformer-bucket) naming.</summary>
    public const string KeyframesEmbeddingKey = "keyframes_abs_pos_embedding";

    /// <summary>Bias whose absence means the video FFN is bias-free, in post-conversion naming.</summary>
    public const string VideoFfnBiasKey = "transformer_blocks.0.ff.net.0.proj.bias";

    /// <summary>Builds the config for a checkpoint. <paramref name="hasKey"/> is queried with post-conversion
    /// transformer-bucket names (the output of <c>LtxVideo2CheckpointConverter</c>).</summary>
    /// <param name="metadata">The safetensors <c>__metadata__</c> entries, or null when the file carries none.</param>
    public static LtxVideo2Config Detect(IReadOnlyDictionary<string, string>? metadata, Func<string, bool> hasKey)
    {
        ArgumentNullException.ThrowIfNull(hasKey);

        LtxVideo2Config config = LtxVideo2Config.V23;
        string version = "unknown";
        bool ffBiasFromConfig = false;

        if (metadata is not null)
        {
            metadata.TryGetValue("model_version", out string? declared);
            version = string.IsNullOrWhiteSpace(declared) ? "unset" : declared;

            if (metadata.TryGetValue("config", out string? configJson) && !string.IsNullOrWhiteSpace(configJson))
            {
                try
                {
                    using JsonDocument doc = JsonDocument.Parse(configJson);
                    if (doc.RootElement.ValueKind == JsonValueKind.Object
                        && doc.RootElement.TryGetProperty("transformer", out JsonElement transformer)
                        && transformer.ValueKind == JsonValueKind.Object)
                    {
                        config = ApplyTransformerConfig(config, transformer);
                        ffBiasFromConfig = transformer.TryGetProperty("ff_bias", out JsonElement ff)
                            && ff.ValueKind is JsonValueKind.True or JsonValueKind.False;
                    }
                }
                catch (JsonException ex)
                {
                    Logs.Error($"LTX-2: checkpoint metadata 'config' is not valid JSON; falling back to key probes.", ex);
                }
            }
        }

        // Key presence is authoritative for the keyframe embedding: a checkpoint either has the parameter or it does
        // not, and reading it wrong is silently wrong output rather than a load failure.
        bool hasKeyframes = hasKey(KeyframesEmbeddingKey);
        // Probe whenever the architecture config didn't state it — "metadata exists" is not the same as "the LTX
        // config was in it". Repacks routinely carry only a `format` entry, and assuming 2.3's biased FFN there would
        // reject the very repacks this detector exists to load.
        bool ffBias = ffBiasFromConfig ? config.FfBias : hasKey(VideoFfnBiasKey);

        config = config with { UseKeyframesAbsPosEmbedding = hasKeyframes, FfBias = ffBias };

        Logs.Info($"LTX-2 variant: model_version={version}, layers={config.NumLayers}, rope={config.RopeType}, " +
            $"ff_bias={config.FfBias}, keyframes_abs_pos={config.UseKeyframesAbsPosEmbedding}");
        return config;
    }

    private static LtxVideo2Config ApplyTransformerConfig(LtxVideo2Config config, JsonElement t)
    {
        config = config with
        {
            NumLayers = Int32Or(t, "num_layers", config.NumLayers),
            NumHeads = Int32Or(t, "num_attention_heads", config.NumHeads),
            HeadDim = Int32Or(t, "attention_head_dim", config.HeadDim),
            InChannels = Int32Or(t, "in_channels", config.InChannels),
            OutChannels = Int32Or(t, "out_channels", config.OutChannels),
            CrossAttentionDim = Int32Or(t, "cross_attention_dim", config.CrossAttentionDim),
            CaptionChannels = Int32Or(t, "caption_channels", config.CaptionChannels),
            AudioNumHeads = Int32Or(t, "audio_num_attention_heads", config.AudioNumHeads),
            AudioHeadDim = Int32Or(t, "audio_attention_head_dim", config.AudioHeadDim),
            AudioInChannels = Int32Or(t, "audio_in_channels", config.AudioInChannels),
            AudioOutChannels = Int32Or(t, "audio_out_channels", config.AudioOutChannels),
            AudioCrossAttentionDim = Int32Or(t, "audio_cross_attention_dim", config.AudioCrossAttentionDim),
            TimestepScaleMultiplier = Int32Or(t, "timestep_scale_multiplier", config.TimestepScaleMultiplier),
            CrossAttnTimestepScaleMultiplier =
                Int32Or(t, "av_ca_timestep_scale_multiplier", config.CrossAttnTimestepScaleMultiplier),
            RopeTheta = SingleOr(t, "positional_embedding_theta", config.RopeTheta),
            NormEps = SingleOr(t, "norm_eps", config.NormEps),
            CrossAttnMod = BoolOr(t, "cross_attention_adaln", config.CrossAttnMod),
            FfBias = BoolOr(t, "ff_bias", config.FfBias),
        };

        if (t.TryGetProperty("rope_type", out JsonElement rope) && rope.ValueKind == JsonValueKind.String)
        {
            string kind = rope.GetString() ?? string.Empty;
            config = config with
            {
                RopeType = kind.Equals("split", StringComparison.OrdinalIgnoreCase)
                    ? LtxVideo2Rope.RopeType.Split
                    : LtxVideo2Rope.RopeType.Interleaved,
            };
        }

        if (TryGetIntArray(t, "positional_embedding_max_pos", out int[] maxPos) && maxPos.Length >= 3)
        {
            config = config with
            {
                RopeBaseNumFrames = maxPos[0],
                RopeBaseHeight = maxPos[1],
                RopeBaseWidth = maxPos[2],
            };
        }

        if (TryGetIntArray(t, "audio_positional_embedding_max_pos", out int[] audioMaxPos) && audioMaxPos.Length >= 1)
            config = config with { AudioPosEmbedMaxPos = audioMaxPos[0] };

        return config;
    }

    private static int Int32Or(JsonElement o, string name, int fallback) =>
        o.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i)
            ? i : fallback;

    private static float SingleOr(JsonElement o, string name, float fallback) =>
        o.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetSingle(out float f)
            ? f : fallback;

    private static bool BoolOr(JsonElement o, string name, bool fallback) =>
        o.TryGetProperty(name, out JsonElement v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean() : fallback;

    private static bool TryGetIntArray(JsonElement o, string name, out int[] values)
    {
        values = [];
        if (!o.TryGetProperty(name, out JsonElement v) || v.ValueKind != JsonValueKind.Array)
            return false;

        List<int> parsed = new(v.GetArrayLength());
        foreach (JsonElement item in v.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int i))
                return false;
            parsed.Add(i);
        }
        values = [.. parsed];
        return true;
    }
}
