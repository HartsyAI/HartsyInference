using System.Text.Json;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Haru's audio codec: the causal VibeVoice acoustic tokenizer (24 kHz audio to 64-d latents at 7.5 Hz and back), built from
/// the <c>codec</c> block of Haru's <c>config.json</c> and its <c>codec.*</c> tensors.</summary>
internal static class HaruCodec
{
    /// <summary>Samples per latent frame the codec must have for the 133 ms frame loop.</summary>
    internal const int FrameSamples = 3200;

    private static readonly HashSet<string> KnownFields = new(StringComparer.Ordinal)
    {
        "latent_dim", "base_channels", "ratios", "encoder_depths", "decoder_depths", "norm_eps", "latent_std",
    };

    /// <summary>Reads the <c>codec</c> block (<c>latent_dim</c>, <c>base_channels</c>, <c>ratios</c>, the two depth lists, <c>norm_eps</c>,
    /// <c>latent_std</c>) into the tokenizer config, and refuses a layout whose frame is not <see cref="FrameSamples"/> long.</summary>
    internal static VibeVoiceTokenizerConfig ConfigFrom(JsonElement codec)
    {
        foreach (JsonProperty field in codec.EnumerateObject())
        {
            if (!KnownFields.Contains(field.Name))
            {
                throw new InvalidDataException($"The Haru codec block has a field '{field.Name}' this loader does not read; refusing to load it against the defaults.");
            }
        }
        int[] ratios = Ints(codec, "ratios");
        int[] encoderDepths = Ints(codec, "encoder_depths");
        int[] decoderDepths = Ints(codec, "decoder_depths");
        int frame = 1;
        foreach (int ratio in ratios)
        {
            frame *= ratio;
        }
        if (frame != FrameSamples)
        {
            throw new InvalidDataException($"The Haru codec ratios give {frame} samples per frame, not {FrameSamples}.");
        }
        if (encoderDepths.Length != ratios.Length + 1 || decoderDepths.Length != encoderDepths.Length)
        {
            throw new InvalidDataException("The Haru codec needs one depth per stage: one more than it has ratios.");
        }
        int baseChannels = codec.GetProperty("base_channels").GetInt32();
        return new VibeVoiceTokenizerConfig
        {
            VaeDim = codec.GetProperty("latent_dim").GetInt32(),
            EncoderNFilters = baseChannels,
            DecoderNFilters = baseChannels,
            EncoderRatios = ratios,
            DecoderRatios = ratios,
            EncoderDepths = string.Join('-', encoderDepths),
            DecoderDepths = string.Join('-', decoderDepths),
            LayerNormEps = (float)codec.GetProperty("norm_eps").GetDouble(),
            // Haru uses latent_std as the fixed spread of its posterior draw (HaruCodecSession), never VibeVoice's per-batch random std,
            // so Sample() on this model is not part of the frame loop.
            FixStd = (float)codec.GetProperty("latent_std").GetDouble(),
        };
    }

    /// <summary>Builds the tokenizer over <paramref name="weights"/> (the checkpoint's tensors, <c>codec.*</c> among them), which must
    /// outlive it. The model reads its tensors through <see cref="HaruCodecWeights.Remap"/>.</summary>
    internal static VibeVoiceAcousticTokenizerModel Create(JsonElement codecConfig, IReadOnlyDictionary<string, Tensor> weights) =>
        Create(ConfigFrom(codecConfig), weights);

    /// <summary><see cref="Create(JsonElement, IReadOnlyDictionary{string, Tensor})"/> from a config already read.</summary>
    internal static VibeVoiceAcousticTokenizerModel Create(VibeVoiceTokenizerConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        VibeVoiceAcousticTokenizerModel model = new(config, HaruCodecWeights.Prefix);
        Dictionary<string, Tensor> remapped = HaruCodecWeights.Remap(weights);
        model.LoadWeights(remapped);
        // The shared loaders read only the keys they expect, so a tensor outside the configured layout (an extra stage, block or down/up
        // layer) would load silently. Every remapped tensor must be one the model took.
        int used = model.EnumerateWeights().Count();
        if (used != remapped.Count)
        {
            throw new InvalidDataException(
                $"The checkpoint has {remapped.Count} codec tensors but the configured codec uses {used}; the layout does not match the codec block.");
        }
        return model;
    }

    private static int[] Ints(JsonElement parent, string name)
    {
        JsonElement array = parent.GetProperty(name);
        int[] values = new int[array.GetArrayLength()];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = array[i].GetInt32();
        }
        return values;
    }
}
