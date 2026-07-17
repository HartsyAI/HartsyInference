namespace HartsyInference.Video.Tokenizers;

/// <summary>Configuration for <see cref="CosmosDvTokenizer"/> (Cosmos-Tokenize1-DV8x16x16-720p). The token-space
/// constants (FSQ levels, compression, latent channels, Haar levels) are fixed by NVIDIA and verified against the
/// research doc; the learned convolutional autoencoder's per-stage channel schedule (<see cref="EncoderChannels"/>
/// / <see cref="DecoderChannels"/>) is resolved at integration from the real <c>encoder.jit</c> weight dump
/// (research-doc Open Q §10) and left null until then.</summary>
public sealed record CosmosDvTokenizerConfig
{
    /// <summary>FSQ per-axis levels. <c>[8,8,8,5,5,5]</c> → 64,000-entry codebook (== AR vocab).</summary>
    public int[] Levels { get; init; } = [8, 8, 8, 5, 5, 5];

    /// <summary>Continuous latent channel count fed to FSQ (= <c>Levels.Length</c> = 6).</summary>
    public int LatentChannels { get; init; } = 6;

    /// <summary>Compression ratio <c>(T, H, W)</c>. DV8x16x16 = 8× temporal, 16× spatial each.</summary>
    public int[] CompressionRatio { get; init; } = [8, 16, 16];

    /// <summary>Number of Haar wavelet levels in the front-end (Cosmos: 2). Implemented as a grouped-conv patcher
    /// (band-major, reflect pad, level-1 first-frame×4) in <see cref="CosmosDvEncoder"/>, not the simple block Haar.</summary>
    public int HaarLevels { get; init; } = 2;

    /// <summary>Base channel width of the conv autoencoder (recovered from <c>encoder.jit</c>): 128.</summary>
    public int BaseChannels { get; init; } = 128;

    /// <summary>Per-stage channel multipliers over conv_in + 3 down stages (recovered): <c>[1,2,4,4]</c>, i.e. the
    /// encoder stage widths are <c>[128, 256, 512, 512]</c> with 2 residual blocks per stage.</summary>
    public int[] ChannelMultipliers { get; init; } = [1, 2, 4, 4];

    /// <summary>Residual blocks per down/up stage (recovered): 2.</summary>
    public int ResBlocksPerStage { get; init; } = 2;

    /// <summary>Encoder stage output channels (post-Haar → pre-FSQ), recovered from <c>encoder.jit</c>:
    /// conv_in 192→128, down.0 →256 (spatiotemporal downsample), down.1 →512 (spatial-only downsample),
    /// down.2 512 (no downsample), mid 512, conv_out 512→16, quant_conv 16→6.</summary>
    public int[] EncoderChannels { get; init; } = [128, 256, 512, 512];

    /// <summary>Decoder stage output channels (post-FSQ → pre-Haar-inverse): post_quant_conv 6→16, conv_in 16→512,
    /// mid 512, up.2/up.1/up.0 512→512→256, conv_out 256→192.</summary>
    public int[] DecoderChannels { get; init; } = [512, 512, 512, 256];

    /// <summary>The shipped DV8x16x16-720p preset (token-space constants + recovered conv schedule).</summary>
    public static CosmosDvTokenizerConfig Dv8x16x16 => new();
}
