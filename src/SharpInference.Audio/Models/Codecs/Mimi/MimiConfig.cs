namespace SharpInference.Audio.Models.Codecs.Mimi;

/// <summary>Configuration for Mimi (Kyutai / Moshi codec). Used by Sesame CSM and the
/// Moshi spoken-LM family. Three architectural pieces stacked vertically:
/// <list type="number">
///   <item>SEANet-style causal encoder (similar to EnCodec24kHz) at 24 kHz → 12.5 Hz</item>
///   <item>Small causal transformer (4 layers, RoPE) over the latent sequence — adds
///         temporal context before quantization</item>
///   <item>Dual-stream RVQ: 1 "semantic" VQ codebook (vocab 2048) + 7 acoustic RVQ
///         codebooks (vocab 2048 each)</item>
/// </list>
/// The transformer-of-codecs in the middle is what distinguishes Mimi from EnCodec —
/// it lets each frame's quantization depend on the full preceding context, dramatically
/// improving reconstruction quality at the same bitrate.</summary>
public sealed record MimiConfig
{
    public int SampleRate { get; init; } = 24_000;
    public int Channels { get; init; } = 1;
    public int EncoderDim { get; init; } = 64;
    public IReadOnlyList<int> EncoderRates { get; init; } = [8, 6, 5, 4];     // 960× downsample → 25 Hz, halved again by upsample inside → 12.5 Hz
    public int LatentDim { get; init; } = 512;

    /// <summary>Transformer-of-codecs hidden dim. Same as <see cref="LatentDim"/> in
    /// Mimi.</summary>
    public int TransformerDim { get; init; } = 512;
    public int TransformerLayers { get; init; } = 8;
    public int TransformerHeads { get; init; } = 8;
    public int TransformerFfnDim { get; init; } = 2_048;
    public float TransformerRopeTheta { get; init; } = 10_000f;

    /// <summary>Acoustic RVQ codebook count. 7 in Mimi (codebook 0 = semantic, codebooks
    /// 1..7 = acoustic residuals).</summary>
    public int AcousticCodebooks { get; init; } = 7;
    public int CodebookSize { get; init; } = 2_048;
    public int CodebookDim { get; init; } = 256;
    public IReadOnlyList<int> ResidualDilations { get; init; } = [1, 1];

    public int FrameRate
    {
        get
        {
            int p = 1;
            for (int i = 0; i < EncoderRates.Count; i++) p *= EncoderRates[i];
            return SampleRate / p;
        }
    }

    /// <summary>Total codebooks = 1 semantic + N acoustic. 8 in the published Mimi
    /// checkpoint.</summary>
    public int TotalCodebooks => 1 + AcousticCodebooks;

    public static MimiConfig Mimi24kHz => new();
}
