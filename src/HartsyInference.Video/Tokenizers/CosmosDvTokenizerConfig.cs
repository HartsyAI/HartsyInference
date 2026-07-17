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

    /// <summary>Number of Haar wavelet levels in the front-end (Cosmos: 2).</summary>
    public int HaarLevels { get; init; } = 2;

    /// <summary>Per-stage output channels of the causal-conv encoder (post-Haar → pre-FSQ). Null until the JIT
    /// weight dump resolves the exact schedule; <see cref="CosmosDvTokenizer.Encode"/> requires loaded weights.</summary>
    public int[]? EncoderChannels { get; init; }

    /// <summary>Per-stage output channels of the causal-conv decoder (post-FSQ → pre-Haar-inverse). Null until
    /// resolved from the JIT weight dump.</summary>
    public int[]? DecoderChannels { get; init; }

    /// <summary>The shipped DV8x16x16-720p preset (token-space constants only; conv schedule pending JIT dump).</summary>
    public static CosmosDvTokenizerConfig Dv8x16x16 => new();
}
