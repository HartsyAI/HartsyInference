namespace HartsyInference.Audio.Models.Codecs.NeuCodec;

/// <summary>Configuration for <see cref="NeuCodecEncoder"/> — the BigCodec acoustic encoder front of NeuCodec.
/// Defaults follow the BigCodec acoustic encoder used by NeuCodec: base width ngf 48, downsample ratios
/// <c>[2,2,4,4,5]</c> (product 320 → 50 Hz at 16 kHz), 3 SnakeBeta residual units per stage with dilations
/// <c>[1,3,9]</c>, and an FSQ head of 8 dims × 4 levels = 65536 codes (mirrors <see cref="NeuCodecConfig"/>'s
/// FSQ settings so the encoder and decoder agree on the codebook). Channel doubling per stage gives a final
/// width of <c>ngf · 2^5 = 1536</c> before <c>final_conv</c>.</summary>
public sealed record NeuCodecEncoderConfig
{
    public int InputSampleRate { get; init; } = 16_000;
    public int FrameRate { get; init; } = 50;

    /// <summary>Base channel width.</summary>
    public int Ngf { get; init; } = 48;

    /// <summary>Downsample ratios (product = total downsample factor, 320×).</summary>
    public IReadOnlyList<int> DownRatios { get; init; } = [2, 2, 4, 4, 5];

    public int ResidualUnitsPerStage { get; init; } = 3;
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 3, 9];
    public int ResidualKernel { get; init; } = 7;

    public int StemKernel { get; init; } = 7;
    public int FinalKernel { get; init; } = 3;

    /// <summary>Channels feeding <c>final_conv</c> output and <c>fc_prior</c> input.</summary>
    public int FcInDim { get; init; } = 1_024;

    /// <summary>FSQ per-dimension levels — 8 dims × 4 levels = 65536 codes (matches the decoder).</summary>
    public IReadOnlyList<int> FsqLevels { get; init; } = [4, 4, 4, 4, 4, 4, 4, 4];

    /// <summary>FSQ code-vector dimension (== FsqLevels.Count).</summary>
    public int FsqDim { get; init; } = 8;

    public static NeuCodecEncoderConfig Default => new();
}
