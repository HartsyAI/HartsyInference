namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>Configuration for SNAC — Multi-Scale Neural Audio Codec (Siuzdak et al.
/// 2024). Used by Orpheus TTS and its derivatives. Architectural overlap with DAC is
/// substantial — same SEANet-style encoder/decoder with snake activations and
/// weight-normed convolutions — with one key extension:
///
/// <para><b>Hierarchical residual VQ.</b> Each codebook operates at a different temporal
/// rate. Earlier codebooks see a heavily downsampled latent (via averaging pool); later
/// codebooks see the latent at progressively higher rates (re-upsampled by repeat).
/// Stride per codebook is configured via <see cref="VqStrides"/>. This lets the codec
/// allocate semantic bits to slow timescales and acoustic detail to fast timescales —
/// reaching higher reconstruction quality at low bitrate than flat-rate RVQ.</para>
///
/// <para>Three official variants:</para>
/// <list type="bullet">
///   <item>SNAC 24 kHz — <c>hubertsiuzdak/snac_24khz</c>, used by Orpheus</item>
///   <item>SNAC 32 kHz — speech-only, slightly larger model</item>
///   <item>SNAC 44.1 kHz — music-focused, more codebooks</item>
/// </list></summary>
public sealed record SnacConfig
{
    public int SampleRate { get; init; } = 24_000;
    public int Channels { get; init; } = 1;
    public int EncoderDim { get; init; } = 48;
    public int DecoderDim { get; init; } = 1_024;

    public IReadOnlyList<int> EncoderRates { get; init; } = [2, 4, 8, 8];
    public IReadOnlyList<int> DecoderRates { get; init; } = [8, 8, 4, 2];

    public int NCodebooks { get; init; } = 3;
    public int CodebookSize { get; init; } = 4_096;
    public int CodebookDim { get; init; } = 8;

    /// <summary>Per-codebook temporal stride. Codebook <c>i</c> processes the latent at
    /// <c>1/VqStrides[i]</c> of the bottleneck frame rate, via avg-pool down on encode
    /// and repeat-interleave up on decode. Bigger stride = coarser timescale. The 24 kHz
    /// speech model uses <c>[4,2,1]</c> → per super-frame the 3 codebooks emit 1/2/4 codes
    /// (the 7-code group Orpheus packs).</summary>
    public IReadOnlyList<int> VqStrides { get; init; } = [4, 2, 1];

    public int ResidualKernelSize { get; init; } = 7;
    public int StemKernelSize { get; init; } = 7;
    public int DecoderFinalKernelSize { get; init; } = 7;
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];

    /// <summary>Convenience — latent dim at the encoder bottleneck. <c>EncoderDim * 2^len(EncoderRates)</c>.</summary>
    public int LatentDim => EncoderDim * (1 << EncoderRates.Count);

    /// <summary>Convenience — bottleneck frame rate in Hz.</summary>
    public int FrameRate
    {
        get
        {
            int product = 1;
            for (int i = 0; i < EncoderRates.Count; i++) product *= EncoderRates[i];
            return SampleRate / product;
        }
    }

    /// <summary>SNAC 24 kHz — Orpheus TTS preset (<c>hubertsiuzdak/snac_24khz</c>).</summary>
    public static SnacConfig Snac24kHz => new();

    /// <summary>SNAC 32 kHz preset.</summary>
    public static SnacConfig Snac32kHz => new()
    {
        SampleRate = 32_000,
        EncoderRates = [2, 4, 8, 8],
        DecoderRates = [8, 8, 4, 2],
        NCodebooks = 4,
        VqStrides = [4, 2, 1, 1],
    };

    /// <summary>SNAC 44.1 kHz preset (music-focused, 3 codebooks).</summary>
    public static SnacConfig Snac44kHz => new()
    {
        SampleRate = 44_100,
        EncoderDim = 64,
        DecoderDim = 1_536,
        EncoderRates = [3, 3, 7, 7],
        DecoderRates = [7, 7, 3, 3],
        NCodebooks = 3,
        VqStrides = [8, 4, 2],
    };
}
