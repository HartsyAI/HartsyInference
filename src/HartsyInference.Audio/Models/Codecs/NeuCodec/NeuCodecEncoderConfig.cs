namespace HartsyInference.Audio.Models.Codecs.NeuCodec;

/// <summary>Configuration for <see cref="NeuCodecEncoder"/> — the X-Codec2 encode path of NeuCodec (HF
/// <c>NeuCodecModel</c>, transformers PR #47143). Two parallel branches at 16 kHz → 50 Hz:
/// an acoustic BigVGAN-style SnakeBeta conv encoder (<c>acoustic_encoder.*</c>) and a semantic
/// Wav2Vec2-BERT conformer (<c>semantic_encoder.*</c>, 16 layers, relative-key attention). Their features
/// are concatenated <c>[semantic, acoustic]</c>, mixed by <c>fc_encoder</c> (2048→2048) and FSQ-quantized
/// (<c>quantizer.project_in</c> 2048→8, levels <c>[4]^8</c> = 65536 codes).</summary>
public sealed record NeuCodecEncoderConfig
{
    public int InputSampleRate { get; init; } = 16_000;
    public int FrameRate { get; init; } = 50;

    // ── Acoustic branch (NeuCodecEncoder) ──
    /// <summary>Base channel width (<c>encoder_hidden_size</c>); stage <c>i</c> has <c>48·2^(i+1)</c> channels.</summary>
    public int AcousticBaseChannels { get; init; } = 48;

    /// <summary>Downsample ratios per stage (product = 320× → 50 Hz at 16 kHz).</summary>
    public IReadOnlyList<int> DownsamplingRatios { get; init; } = [2, 2, 4, 4, 5];

    public int AcousticResidualKernel { get; init; } = 7;
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];

    /// <summary>Kaiser-sinc anti-aliasing (BigVGAN Activation1d) around every SnakeBeta: up/down ratio 2, kernel 12.</summary>
    public int AntiAliasKernel { get; init; } = 12;
    public int AntiAliasRatio { get; init; } = 2;

    // ── Semantic branch (Wav2Vec2-BERT conformer) ──
    public int HiddenSize { get; init; } = 1_024;
    public int SemanticLayers { get; init; } = 16;
    public int SemanticHeads { get; init; } = 16;
    public int SemanticHeadDim { get; init; } = 64;
    public int SemanticIntermediate { get; init; } = 4_096;
    public int ConvDepthwiseKernel { get; init; } = 31;
    public int LeftMaxPosition { get; init; } = 64;
    public int RightMaxPosition { get; init; } = 8;
    public int FeatureProjInputDim { get; init; } = 160;   // 80 mel bins × stride-2 stacking
    public int NumMelBins { get; init; } = 80;
    public float LayerNormEps { get; init; } = 1e-5f;

    // ── Fusion + quantizer ──
    /// <summary>fc_encoder / quantizer.project_in input dim = acoustic 1024 + semantic 1024.</summary>
    public int FusionDim { get; init; } = 2_048;

    /// <summary>FSQ per-dimension levels — 8 dims × 4 levels = 65536 codes (matches the decoder).</summary>
    public IReadOnlyList<int> FsqLevels { get; init; } = [4, 4, 4, 4, 4, 4, 4, 4];
    public int FsqDim { get; init; } = 8;

    public static NeuCodecEncoderConfig Default => new();
}
