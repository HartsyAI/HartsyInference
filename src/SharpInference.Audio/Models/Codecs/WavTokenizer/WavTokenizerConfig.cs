namespace SharpInference.Audio.Models.Codecs.WavTokenizer;

/// <summary>Configuration for WavTokenizer (Ji et al. 2024). Single-codebook neural
/// audio codec with a Vocos-style iSTFT decoder. Used by Orpheus forks and several
/// research-grade single-token TTS systems where the "one token per frame" property
/// simplifies the LM-side modeling.
///
/// <para>Encoder: SEANet-style (Conv1d + snake activations + downsample stages) at
/// 320× downsample → 75 Hz at 24 kHz input. Single codebook of 4096 entries — 12 bits
/// per frame, 900 bits/s = 0.9 kbps.</para>
///
/// <para>Decoder: ConvNeXt blocks + linear projection to mag/phase + iSTFT
/// (frequency-domain vocoder, faster than time-domain transposed convs at this rate).</para></summary>
public sealed record WavTokenizerConfig
{
    public int SampleRate { get; init; } = 24_000;
    public int Channels { get; init; } = 1;
    public int EncoderDim { get; init; } = 64;
    public IReadOnlyList<int> EncoderRates { get; init; } = [8, 5, 4, 2];
    public int LatentDim { get; init; } = 512;
    public int CodebookSize { get; init; } = 4_096;
    public int CodebookDim { get; init; } = 8;
    public int ResidualKernelSize { get; init; } = 7;
    public int StemKernelSize { get; init; } = 7;
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];

    /// <summary>iSTFT n_fft. 1280 for the 24 kHz WavTokenizer head (matches Vocos).</summary>
    public int NFft { get; init; } = 1_280;
    public int HopLength { get; init; } = 320;
    public int HeadDim { get; init; } = 768;
    public int HeadConvNeXtBlocks { get; init; } = 8;
    public int HeadFfnRatio { get; init; } = 3;

    public int FrameRate
    {
        get
        {
            int p = 1;
            for (int i = 0; i < EncoderRates.Count; i++) p *= EncoderRates[i];
            return SampleRate / p;
        }
    }

    public static WavTokenizerConfig WavTokenizer24kHz => new();
}
