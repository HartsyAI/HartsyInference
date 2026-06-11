namespace SharpInference.Audio.Models.Codecs.Oobleck;

/// <summary>Configuration for the Oobleck waveform VAE (Stability AI's Stable-Audio autoencoder,
/// diffusers <c>AutoencoderOobleck</c>). Field names and defaults mirror the diffusers config so a
/// checkpoint's <c>config.json</c> maps 1:1. Used by Stable Audio Open (44.1 kHz, ratios 2·4·4·8·8)
/// and ACE-Step 1.5 (48 kHz, ratios 2·4·4·6·10 — see <see cref="AceStep15"/>).</summary>
public record OobleckConfig
{
    /// <summary>Encoder stem width; also the encoder output channels (= 2 × latent dim, mean+scale).</summary>
    public int EncoderHiddenSize { get; init; } = 128;

    /// <summary>Per-stage encoder strides; the decoder upsamples with the reversed list.</summary>
    public int[] DownsamplingRatios { get; init; } = [2, 4, 4, 8, 8];

    /// <summary>Hidden-width multipliers per stage (a leading implicit 1 is prepended at build time).</summary>
    public int[] ChannelMultiples { get; init; } = [1, 2, 4, 8, 16];

    /// <summary>Decoder stem width (final pre-output channel count).</summary>
    public int DecoderChannels { get; init; } = 128;

    /// <summary>Latent dimensionality (the decoder's input channels).</summary>
    public int DecoderInputChannels { get; init; } = 64;

    /// <summary>1 = mono, 2 = stereo.</summary>
    public int AudioChannels { get; init; } = 2;

    /// <summary>Waveform sample rate in Hz.</summary>
    public int SamplingRate { get; init; } = 44100;

    /// <summary>Samples per latent frame = product of <see cref="DownsamplingRatios"/>.</summary>
    public int HopLength
    {
        get
        {
            int hop = 1;
            foreach (int r in DownsamplingRatios) hop *= r;
            return hop;
        }
    }

    /// <summary>Stable Audio Open 1.0 — stereo 44.1 kHz, 64-d latents @ ~21.5 Hz (hop 2048).</summary>
    public static OobleckConfig StableAudioOpen => new();

    /// <summary>ACE-Step 1.5 — stereo 48 kHz, 64-d latents @ 25 Hz (hop 1920). Matches the
    /// <c>vae/config.json</c> in <c>ACE-Step/Ace-Step1.5</c>.</summary>
    public static OobleckConfig AceStep15 => new()
    {
        DownsamplingRatios = [2, 4, 4, 6, 10],
        SamplingRate = 48000,
    };
}
