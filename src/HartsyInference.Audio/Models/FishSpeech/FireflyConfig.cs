namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Firefly-gan-vq codec configuration (Fish-Speech 1.5). Covers the grouped-residual FSQ quantizer
/// (<c>DownsampleFiniteScalarQuantize</c>) and the SiLU HiFi-GAN generator. Defaults match the published
/// <c>firefly-gan-vq-fsq-8x1024-21hz-generator</c> checkpoint.</summary>
public sealed record FireflyConfig
{
    /// <summary>Per-book FSQ levels — product is the per-codebook vocabulary (<c>8·5·5·5 = 1000</c>).</summary>
    public int[] FsqLevels { get; init; } = [8, 5, 5, 5];

    /// <summary>ConvNeXt down/up-sample factors of the quantizer (frames ×4 on decode).</summary>
    public int[] QuantizerUpsampleFactors { get; init; } = [2, 2];

    /// <summary>HiFi-GAN generator initial channel count (halves per upsample stage).</summary>
    public int UpsampleInitialChannel { get; init; } = 512;

    /// <summary>HiFi-GAN per-stage upsample strides (product = waveform hop per latent frame).</summary>
    public int[] UpsampleRates { get; init; } = [8, 8, 2, 2, 2];

    /// <summary>HiFi-GAN per-stage ConvTranspose1d kernel sizes.</summary>
    public int[] UpsampleKernelSizes { get; init; } = [16, 16, 4, 4, 4];

    /// <summary>MRF ResBlock1 kernel sizes.</summary>
    public int[] ResBlockKernelSizes { get; init; } = [3, 7, 11];

    /// <summary>MRF ResBlock1 per-kernel dilation triples.</summary>
    public int[][] ResBlockDilations { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    public int SampleRate { get; init; } = 44_100;

    /// <summary>Number of residual groups in the grouped-residual FSQ quantizer.</summary>
    public int NGroups { get; init; } = 8;

    /// <summary>Number of codebooks per quantizer group.</summary>
    public int QuantizerNCodebooks { get; init; } = 1;

    /// <summary>Quantizer input/projection dimension.</summary>
    public int QuantizerInputDim { get; init; } = 512;

    /// <summary>Number of mel channels feeding the generator backbone.</summary>
    public int NumMels { get; init; } = 512;

    /// <summary>Mel/STFT hop length (waveform samples per latent frame).</summary>
    public int HopLength { get; init; } = 512;

    /// <summary>Pre-convolution (input stem) kernel size.</summary>
    public int PreConvKernelSize { get; init; } = 13;

    /// <summary>Post-convolution (output stem) kernel size.</summary>
    public int PostConvKernelSize { get; init; } = 13;

    public static FireflyConfig V1_5 => new();
}
