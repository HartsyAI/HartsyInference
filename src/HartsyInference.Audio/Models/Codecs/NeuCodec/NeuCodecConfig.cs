namespace HartsyInference.Audio.Models.Codecs.NeuCodec;

/// <summary>Configuration for NeuCodec (Neuphonic) — a single-codebook 0.8 kbps neural audio codec used by
/// NeuTTS Air. 16 kHz in / 24 kHz out, 50 Hz token rate, one FSQ codebook of <c>4^8 = 65536</c> codes.
/// The decoder (the path TTS needs) is Vocos-style: FSQ de-quant → 1024-d backbone (Conv embed + ResNet
/// pre-net + 12 RoPE transformer blocks + ResNet post-net) → iSTFT head (no transposed-conv upsampling;
/// the 480-sample hop on the 24 kHz grid does the rate expansion). See
/// <c>docs/Research/KYUTAI_DSM_ARCHITECTURE.md</c>'s sibling notes; lineage is X-Codec2 / BigCodec.
///
/// <para><b>Reuse:</b> FSQ index→codes via <c>Fsq.Dequantize</c>; iSTFT head via <c>IStft.Apply</c>; conv/
/// norm/attention via <c>IBackend</c> ops. Net-new: the transformer backbone + ResNet blocks.</para></summary>
public sealed record NeuCodecConfig
{
    public int InputSampleRate { get; init; } = 16_000;
    public int SampleRate { get; init; } = 24_000;        // decoder output rate
    public int HopLength { get; init; } = 480;
    public int FrameRate { get; init; } = 50;

    /// <summary>FSQ per-dimension levels — 8 dims × 4 levels = 65536 codes.</summary>
    public IReadOnlyList<int> FsqLevels { get; init; } = [4, 4, 4, 4, 4, 4, 4, 4];

    /// <summary>FSQ code-vector dimension (== FsqLevels.Count).</summary>
    public int FsqDim { get; init; } = 8;

    /// <summary>Quantizer project-out dim (FSQ 8 → 2048), then fc_post_a 2048 → backbone dim.</summary>
    public int QuantizerDim { get; init; } = 2_048;

    public int BackboneDim { get; init; } = 1_024;
    public int Depth { get; init; } = 12;
    public int NumHeads { get; init; } = 16;
    public int HeadDim => BackboneDim / NumHeads;          // 64
    public int MlpDim { get; init; } = 4_096;
    public int EmbedKernel { get; init; } = 7;
    public int ResnetKernel { get; init; } = 3;
    public int GroupNormGroups { get; init; } = 32;
    public int PriorResnetBlocks { get; init; } = 2;
    public int PostResnetBlocks { get; init; } = 2;
    public float NormEps { get; init; } = 1e-6f;
    public float RopeTheta { get; init; } = 10_000f;

    public int NFft { get; init; } = 1_920;               // hop_length * 4
    public int Win { get; init; } = 1_920;

    /// <summary>Single codebook of 65536 codes.</summary>
    public int CodebookSize { get; init; } = 65_536;

    public static NeuCodecConfig Default => new();
}
