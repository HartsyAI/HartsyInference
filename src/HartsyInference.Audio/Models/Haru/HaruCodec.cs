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

    /// <summary>Reads the <c>codec</c> block (<c>latent_dim</c>, <c>base_channels</c>, <c>ratios</c>, the two depth lists, <c>norm_eps</c>,
    /// <c>latent_std</c>) into the tokenizer config, and refuses a layout whose frame is not <see cref="FrameSamples"/> long.</summary>
    internal static VibeVoiceTokenizerConfig ConfigFrom(JsonElement codec)
    {
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
            FixStd = (float)codec.GetProperty("latent_std").GetDouble(),
        };
    }

    /// <summary>Builds the tokenizer over <paramref name="weights"/> (the checkpoint's tensors, <c>codec.*</c> among them), which must
    /// outlive it. The model reads its tensors through <see cref="HaruCodecWeights.Remap"/>.</summary>
    internal static VibeVoiceAcousticTokenizerModel Create(JsonElement codecConfig, IReadOnlyDictionary<string, Tensor> weights)
    {
        VibeVoiceAcousticTokenizerModel model = new(ConfigFrom(codecConfig), HaruCodecWeights.Prefix);
        model.LoadWeights(HaruCodecWeights.Remap(weights));
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
