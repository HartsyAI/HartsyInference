namespace HartsyInference.Audio.Models.Codecs.Dac;

/// <summary>Configuration for Descript Audio Codec (<c>descript-audio-codec</c>,
/// Kumar et al. 2023). Defaults match the 44.1 kHz checkpoint. Use the static factories
/// for the 24 kHz and 16 kHz variants — they share the same architecture but differ in
/// stride list, codebook count, and consumed bandwidth.
///
/// <para>Architectural differences from <see cref="EnCodec.EnCodecConfig"/>:</para>
/// <list type="bullet">
///   <item>Snake activations (per-channel learnable alpha) instead of ELU</item>
///   <item>No LSTM bottleneck — pure ConvNeXt-style residual stack</item>
///   <item>RVQ uses L2-normalized cosine codebooks with per-codebook in/out 1×1 projections to a smaller <c>codebook_dim</c></item>
///   <item>Symmetric (non-causal) padding everywhere — DAC is offline-only by design</item>
///   <item>Decoder appends a <c>tanh</c> on the output so reconstructions stay in <c>[-1, 1]</c></item>
/// </list>
///
/// <para>Used by IndexTTS, Higgs Audio (acoustic branch), Spark-TTS (HiFi-GAN wave-gen
/// runs against the 16 kHz variant), Orpheus (via SNAC, but the codebook layout is
/// near-identical to DAC).</para></summary>
public sealed record DacConfig
{
    /// <summary>Source sample rate. 44100 / 24000 / 16000 for the three official
    /// checkpoints.</summary>
    public int SampleRate { get; init; } = 44_100;

    /// <summary>Audio channels — always 1 (mono) for the published DAC checkpoints.</summary>
    public int Channels { get; init; } = 1;

    /// <summary>Encoder stem filter count. Always 64 in the published checkpoints.
    /// Channel widths double per stage: 64 → 128 → 256 → 512 → 1024.</summary>
    public int EncoderDim { get; init; } = 64;

    /// <summary>Decoder initial filter count. Always 1536 in the published checkpoints.
    /// Channels halve per stage: 1536 → 768 → 384 → 192 → 96 (then projected to 1 by
    /// the final conv).</summary>
    public int DecoderDim { get; init; } = 1_536;

    /// <summary>Per-stage downsample factors (encoder-forward order). Product equals the
    /// total downsample / upsample ratio.</summary>
    public IReadOnlyList<int> EncoderRates { get; init; } = [2, 4, 8, 8];

    /// <summary>Per-stage upsample factors (decoder-forward order — mirror of
    /// <see cref="EncoderRates"/>).</summary>
    public IReadOnlyList<int> DecoderRates { get; init; } = [8, 8, 4, 2];

    /// <summary>Number of residual VQ codebooks. 9 (44.1 kHz, ~8 kbps), 32 (24 kHz),
    /// 12 (16 kHz). Each codebook has 1024 entries → ~10 bits/codebook, ~860 bits/s per
    /// codebook at 86 Hz.</summary>
    public int NCodebooks { get; init; } = 9;

    /// <summary>Codebook size — 1024 across all official checkpoints.</summary>
    public int CodebookSize { get; init; } = 1_024;

    /// <summary>Codebook dimension — small (8) so the RVQ projects down before the
    /// nearest-neighbor lookup. The actual latent dimension is computed from the encoder.</summary>
    public int CodebookDim { get; init; } = 8;

    /// <summary>Conv kernel size inside the residual units. Fixed at 7.</summary>
    public int ResidualKernelSize { get; init; } = 7;

    /// <summary>Stem kernel size. Fixed at 7.</summary>
    public int StemKernelSize { get; init; } = 7;

    /// <summary>Final-projection kernel size. 3 for encoder projection, 7 for decoder
    /// final conv.</summary>
    public int EncoderProjKernelSize { get; init; } = 3;

    /// <summary>Decoder final-conv kernel size. 7 in the published checkpoints.</summary>
    public int DecoderFinalKernelSize { get; init; } = 7;

    /// <summary>Per-residual-unit dilations applied within an Encoder/DecoderBlock.
    /// Always <c>[1, 3, 9]</c> — three blocks with exponentially-growing receptive fields.</summary>
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];

    /// <summary>Explicit per-stage transposed-conv kernel sizes. Null (default) derives <c>2 × stride</c> per
    /// stage with descript's <c>output_padding = stride % 2</c>. Provide explicit sizes (e.g. Spark BiCodec
    /// <c>[16, 11, 8, 4]</c>) to use the <c>padding = (kernel - stride) / 2</c>, no-output-padding convention.</summary>
    public IReadOnlyList<int>? DecoderKernelSizes { get; init; } = null;

    /// <summary>How the decoder's transposed-conv <c>weight_norm</c> is composed. Default (false) uses the
    /// per-output-channel convention (EnCodec/SeaNet, <c>weight_g</c> sized C_out). Set true for the
    /// descript-audio-codec default (and Spark-TTS BiCodec), whose <c>WNConvTranspose1d</c> norms over
    /// dim 0 with <c>weight_g</c> sized <c>[C_in, 1, 1]</c>.</summary>
    public bool TransposeWeightNormDim0 { get; init; } = false;

    /// <summary>Convenience — the latent dimension at the encoder output (= bottleneck
    /// channels = <c>EncoderDim * 2^len(EncoderRates)</c>). 64 * 16 = 1024 for the
    /// published configs.</summary>
    public int LatentDim => EncoderDim * (1 << EncoderRates.Count);

    /// <summary>Convenience — the latent frame rate (Hz).</summary>
    public int FrameRate
    {
        get
        {
            int product = 1;
            for (int i = 0; i < EncoderRates.Count; i++) product *= EncoderRates[i];
            return SampleRate / product;
        }
    }

    /// <summary>DAC 44.1 kHz — <c>descript/dac_44khz</c>, used by IndexTTS 1.5/2 and
    /// Higgs Audio's acoustic branch.</summary>
    public static DacConfig Dac44kHz => new();

    /// <summary>DAC 24 kHz — <c>descript/dac_24khz</c>, lower-bandwidth variant.</summary>
    public static DacConfig Dac24kHz => new()
    {
        SampleRate = 24_000,
        EncoderRates = [2, 4, 5, 8],
        DecoderRates = [8, 5, 4, 2],
        NCodebooks = 32,
    };

    /// <summary>DAC 16 kHz — <c>descript/dac_16khz</c>, used by Spark-TTS HiFi-GAN
    /// wave-gen.</summary>
    public static DacConfig Dac16kHz => new()
    {
        SampleRate = 16_000,
        EncoderRates = [2, 4, 5, 8],
        DecoderRates = [8, 5, 4, 2],
        NCodebooks = 12,
    };
}
