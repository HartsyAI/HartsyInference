namespace HartsyInference.Audio.Models.Codecs.WavTokenizer;

/// <summary>Configuration for WavTokenizer (Ji et al. 2024) — a single-codebook neural audio codec with a Vocos-style iSTFT decoder, used by Orpheus forks and several research-grade single-token TTS systems.</summary>
/// <remarks>
/// The "one token per frame" property simplifies the LM-side modeling.
///
/// <para>Encoder: SEANet-style (Conv1d + snake activations + downsample stages) at
/// 320× downsample → 75 Hz at 24 kHz input. Single codebook of 4096 entries — 12 bits
/// per frame, 900 bits/s = 0.9 kbps.</para>
///
/// <para>Decoder: ConvNeXt blocks + linear projection to mag/phase + iSTFT
/// (frequency-domain vocoder, faster than time-domain transposed convs at this rate).</para></remarks>
public sealed record WavTokenizerConfig
{
    public int SampleRate { get; init; } = 24_000;
    public int Channels { get; init; } = 1;
    public int EncoderDim { get; init; } = 32;
    public IReadOnlyList<int> EncoderRates { get; init; } = [8, 5, 4, 2];
    public int LatentDim { get; init; } = 512;
    public int CodebookSize { get; init; } = 4_096;

    /// <summary>VQ codebook dimension. 512 (no factorized codebook; RVQ runs at LatentDim).</summary>
    public int CodebookDim { get; init; } = 512;
    public int ResidualKernelSize { get; init; } = 3;
    public int StemKernelSize { get; init; } = 7;

    /// <summary>EnCodec-style residual dilations (dilation_base 2, n_residual_layers 1).</summary>
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 2];

    /// <summary>Number of LSTM layers in the SEANet encoder.</summary>
    public int EncoderLstmLayers { get; init; } = 2;

    /// <summary>Number of AdaNorm conditioning embeddings in the ConvNeXt decoder head.</summary>
    public int AdaNormNumEmbeddings { get; init; } = 4;

    /// <summary>iSTFT n_fft. 1280 for the 24 kHz WavTokenizer head (matches Vocos).</summary>
    public int NFft { get; init; } = 1_280;
    public int HopLength { get; init; } = 320;
    public int HeadDim { get; init; } = 768;
    public int HeadConvNeXtBlocks { get; init; } = 12;
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

    /// <summary>40 tokens/s variant (24 kHz / 600 hop): EncoderRates [6,5,5,4], 2400-point iSTFT, 600 hop, same ConvNeXt backbone dims as the default.</summary>
    public static WavTokenizerConfig Frame40 => new()
    {
        EncoderRates = [6, 5, 5, 4],
        NFft = 2_400,
        HopLength = 600,
    };
}
