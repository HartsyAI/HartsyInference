namespace HartsyInference.Diffusion.Models.Music;

/// <summary>Geometry and latent statistics of the MiniMax-H3 audio VAE decoder. Defaults are the shipped
/// <c>audio_vae/config.json</c> + <c>config.yaml</c> + <c>metadata.json</c> triple: 32 kHz, 32 latent channels,
/// <c>latent_dim</c> 2048, <c>decoder_dim</c> 1024, decoder rates <c>[5,5,2,2,2,2,2]</c> (∏ = 800 samples per latent
/// frame → a 40 Hz latent rate). Every field is overridable so tests can build a miniature decoder.</summary>
public sealed record MiniMaxH3AudioVaeConfig
{
    /// <summary>Waveform sample rate the BigVGAN decoder emits.</summary>
    public int SampleRate { get; init; } = 32000;

    /// <summary>Channel count of the DiT-side latent (<c>vae_latent_channels</c>).</summary>
    public int LatentChannels { get; init; } = 32;

    /// <summary>Width <c>dec_in_proj</c> lifts the latent to — the BigVGAN <c>num_mels</c> input (<c>latent_dim</c>).</summary>
    public int DecoderInputChannels { get; init; } = 2048;

    /// <summary>BigVGAN <c>upsample_initial_channel</c>; halves after every upsample stage.</summary>
    public int DecoderDim { get; init; } = 1024;

    public int[] UpsampleRates { get; init; } = [5, 5, 2, 2, 2, 2, 2];

    public int[] UpsampleKernels { get; init; } = [9, 9, 4, 4, 4, 4, 4];

    public int[] ResblockKernels { get; init; } = [3, 7, 11];

    public int[][] ResblockDilations { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>Per-channel latent mean; decode denormalizes with <c>latent · std + mean</c>.</summary>
    public float[] LatentsMean { get; init; } =
    [
        -0.020211687f, 0.38764665f, -0.043982798f, -0.28591515f, 0.081796862f, -0.35782641f, 0.040623810f,
        -0.015525345f, -0.22336248f, 0.18210068f, 0.29417788f, -0.079011676f, -0.056815073f, -0.36990282f,
        -0.31616316f, 0.59059514f, -0.052139568f, 0.013673160f, -0.036916479f, 0.097326607f, -0.33946623f,
        -0.30685678f, -0.24504599f, -0.034698524f, 0.028680322f, -0.21217779f, -0.16782632f, 0.32212879f,
        -0.12230559f, 0.43566049f, -0.050259920f, 0.39792584f,
    ];

    /// <summary>Per-channel latent standard deviation; decode denormalizes with <c>latent · std + mean</c>.</summary>
    public float[] LatentsStd { get; init; } =
    [
        1.6895524f, 2.7626373f, 1.7945344f, 1.6801682f, 1.6390227f, 2.7788298f, 1.7659090f, 1.6199758f,
        2.6336526f, 1.8539357f, 2.5056498f, 1.8110192f, 1.9579658f, 1.6685498f, 1.4922469f, 3.2986702f,
        1.9491804f, 1.8720003f, 1.8334080f, 1.6488070f, 1.6176958f, 1.9131449f, 1.5695245f, 1.6943660f,
        1.8318421f, 1.5540637f, 1.9344930f, 1.5991982f, 1.7180460f, 1.6307219f, 1.8661226f, 1.5613768f,
    ];

    /// <summary>DAC encoder stem width; each stage doubles it.</summary>
    public int EncoderDim { get; init; } = 64;

    /// <summary>Per-stage encoder strides; the product is the hop, so it must mirror <see cref="UpsampleRates"/>.</summary>
    public int[] EncoderRates { get; init; } = [2, 4, 4, 5, 5];

    /// <summary>Heads in the posterior head's causal attention.</summary>
    public int AttnHeads { get; init; } = 8;

    /// <summary>Hidden-width multiplier of the posterior head's gated MLP.</summary>
    public int MlpRatio { get; init; } = 2;

    public float NormEps { get; init; } = 1e-5f;

    /// <summary>Waveform samples consumed per latent frame (∏ <see cref="EncoderRates"/>); the encode-side hop.</summary>
    public int EncoderHopLength => EncoderRates.Aggregate(1, (a, r) => a * r);

    /// <summary>Waveform samples produced per latent frame (∏ <see cref="UpsampleRates"/>).</summary>
    public int SamplesPerLatentFrame => UpsampleRates.Aggregate(1, (a, r) => a * r);
}
