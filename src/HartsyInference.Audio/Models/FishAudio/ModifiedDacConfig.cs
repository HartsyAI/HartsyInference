namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Decoder-side configuration of fish-speech's ModifiedDAC codec (<c>configs/modded_dac_vq.yaml</c>): a
/// 4096-entry semantic codebook plus nine 1024-entry residual codebooks, an 8-layer window-limited causal transformer,
/// a 4× causal upsampler and a causal DAC decoder.</summary>
public sealed record ModifiedDacConfig
{
    public int SampleRate { get; init; } = 44_100;
    public int LatentDim { get; init; } = 1_024;
    public int CodebookDim { get; init; } = 8;
    public int NumResidualCodebooks { get; init; } = 9;
    public int SemanticCodebookSize { get; init; } = 4_096;
    public int ResidualCodebookSize { get; init; } = 1_024;
    public int[] UpsampleFactors { get; init; } = [2, 2];
    public int[] DownsampleFactors { get; init; } = [2, 2];
    public int DecoderDim { get; init; } = 1_536;
    public int[] DecoderRates { get; init; } = [8, 8, 4, 2];
    public int[] ResidualDilations { get; init; } = [1, 3, 9];
    public int ConvNeXtKernel { get; init; } = 7;

    // Encoder: conv(1→EncoderDim) → one EncoderBlock per rate (channels double each block) → Snake → conv(→LatentDim).
    public int EncoderDim { get; init; } = 64;
    public int[] EncoderRates { get; init; } = [2, 4, 8, 8];
    public int[] EncoderTransformerLayers { get; init; } = [0, 0, 0, 4];
    public int EncoderTransformerWindow { get; init; } = 512;

    // post_module: WindowLimitedTransformer(causal, window 128), dim == LatentDim so no in/out projection.
    public int TransformerLayers { get; init; } = 8;
    public int TransformerHeads { get; init; } = 16;
    public int TransformerHeadDim { get; init; } = 64;
    public int TransformerIntermediate { get; init; } = 3_072;
    public int TransformerWindow { get; init; } = 128;
    public float TransformerRopeBase { get; init; } = 10_000f;
    public float TransformerNormEps { get; init; } = 1e-5f;

    public int TotalCodebooks => 1 + NumResidualCodebooks;

    /// <summary>Audio samples per code frame on the encode side (encoder rates × downsample factors).</summary>
    public int EncodeSamplesPerFrame => EncoderRates.Aggregate(1, (a, b) => a * b) * DownsampleFactors.Aggregate(1, (a, b) => a * b);

    /// <summary>Audio samples produced per code frame (upsample factors × decoder rates).</summary>
    public int SamplesPerFrame => UpsampleFactors.Aggregate(1, (a, b) => a * b) * DecoderRates.Aggregate(1, (a, b) => a * b);

    public static ModifiedDacConfig S2 => new();
}
