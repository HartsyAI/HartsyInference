namespace HartsyInference.Audio.Models.Codecs.Oobleck;

/// <summary>Configuration for the Oobleck waveform VAE (Stability AI's Stable-Audio autoencoder, diffusers <c>AutoencoderOobleck</c>).</summary>
/// <remarks>Field names and defaults mirror the diffusers config so a checkpoint's <c>config.json</c> maps 1:1.
/// Used by Stable Audio Open (44.1 kHz, ratios 2·4·4·8·8) and ACE-Step 1.5 (48 kHz, ratios 2·4·4·6·10 — see
/// <see cref="AceStep15"/>).</remarks>
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

    /// <summary>Samples a decode of <paramref name="frames"/> latent frames actually produces, which is
    /// <see cref="HopLength"/> per frame <b>minus a fixed edge loss</b> whenever any ratio is odd.</summary>
    /// <remarks>A transpose conv here runs <c>k = 2s</c> with <c>padding = ceil(s/2)</c>, so it emits
    /// <c>sL</c> for an even stride but <c>sL - 1</c> for an odd one, and everything above that stage multiplies the
    /// shortfall. <see cref="Yue2"/>'s stride of 5 sits under a further 64× of upsampling and costs exactly 64
    /// samples; every all-even config is exact. The loss is a constant, not a function of length, so it comes off the
    /// end of a decode once — which is why tiling must ask this rather than multiply by the hop.</remarks>
    public long DecodedLength(long frames)
    {
        long length = frames;
        // The decoder upsamples with the reversed list.
        for (int i = DownsamplingRatios.Length - 1; i >= 0; i--)
        {
            int stride = DownsamplingRatios[i], kernel = 2 * stride, padding = (stride + 1) / 2;
            length = (length - 1) * stride - 2L * padding + kernel;
        }
        return length;
    }

    /// <summary>Samples per latent frame = product of <see cref="DownsamplingRatios"/>. This is the tiling and
    /// alignment ratio; for an exact output length use <see cref="DecodedLength"/>.</summary>
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

    /// <summary>ACE-Step 1.5 — stereo 48 kHz, 64-d latents @ 25 Hz (hop 1920). Matches the <c>vae/config.json</c> in <c>ACE-Step/Ace-Step1.5</c>.</summary>
    public static OobleckConfig AceStep15 => new()
    {
        DownsamplingRatios = [2, 4, 4, 6, 10],
        SamplingRate = 48000,
    };

    /// <summary>YuE2 — stereo 48 kHz, 64-d latents @ 25 Hz (hop 1920), but over <b>six</b> stages with a narrow
    /// 64-channel stem instead of the usual five and 128. Derived from the decoder's transposed-conv kernel widths
    /// in <c>m-a-p/YuE2-Vae</c> (12, 10, 8, 8, 4, 4 = 2·stride).</summary>
    /// <remarks>The stride-5 stage is the first odd ratio any Oobleck checkpoint in this tree has used. Upstream
    /// leaves <c>output_padding</c> at zero, so that stage emits <c>5L − 1</c> rather than <c>5L</c> frames and a
    /// whole decode is slightly shorter than <c>frames × 1920</c> — 8 frames give 15,296 samples, not 15,360. Our
    /// decoder's existing length rule already computes this; ComfyUI's port adds <c>output_padding = stride % 2</c>
    /// and does not match the model's own decoder.</remarks>
    public static OobleckConfig Yue2 => new()
    {
        EncoderHiddenSize = 64,
        DownsamplingRatios = [2, 2, 4, 4, 5, 6],
        ChannelMultiples = [1, 2, 4, 8, 16, 32],
        DecoderChannels = 64,
        SamplingRate = 48000,
    };
}
