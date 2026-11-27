namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Configuration of the HunyuanVideo-1.5 causal VAE (<c>AutoencoderKLConv3D</c> in MovieRoll's <c>causal_vae.py</c>, the same
/// weights as HunyuanVideo-1.5). Unlike the 13B HunyuanVideo VAE (<see cref="HunyuanVideoVaeConfig"/>) it uses RMS norm instead of GroupNorm,
/// DCAE shuffle resamplers with channel-shortcut residuals, a 32-channel latent and 16× spatial / 4× temporal compression.
/// Defaults are the published <c>vae/config.json</c>; <see cref="ScalingFactor"/> must be confirmed against that file before a real-weight run.</summary>
public sealed record HunyuanVideo15VaeConfig
{
    /// <summary>RGB input channels.</summary>
    public int InChannels { get; init; } = 3;

    /// <summary>RGB output channels.</summary>
    public int OutChannels { get; init; } = 3;

    /// <summary>Latent channels. The encoder emits <c>2 ×</c> this (mean, then logvar); the decoder consumes this many.</summary>
    public int LatentChannels { get; init; } = 32;

    /// <summary>Per-level channel counts, encoder order. The decoder runs them reversed.</summary>
    public int[] BlockOutChannels { get; init; } = [128, 256, 512, 1024, 1024];

    /// <summary>Resnets per encoder level (the decoder runs one more per level).</summary>
    public int LayersPerBlock { get; init; } = 2;

    /// <summary>Spatial compression (<c>ffactor_spatial</c>). Must equal <c>2^(BlockOutChannels.Length − 1)</c>.</summary>
    public int SpatialCompression { get; init; } = 16;

    /// <summary>Temporal compression (<c>ffactor_temporal</c>). MovieRoll's frame arithmetic requires 4.</summary>
    public int TemporalCompression { get; init; } = 4;

    /// <summary>Latent scale: the pipeline divides a latent by this before decoding (MovieRoll <c>generate.py</c>). Confirm against <c>vae/config.json</c>.</summary>
    public float ScalingFactor { get; init; } = 1.03682f;

    /// <summary>RMS-norm epsilon (<c>F.normalize</c> default).</summary>
    public float NormEps { get; init; } = 1e-12f;

    /// <summary>The published HunyuanVideo-1.5 VAE configuration.</summary>
    public static HunyuanVideo15VaeConfig Default => new();

    /// <summary>Encoder level <paramref name="level"/> downsamples space (the first <c>log2(SpatialCompression)</c> levels).</summary>
    public bool EncoderSpatialDownsample(int level) => level < Log2(SpatialCompression);

    /// <summary>Encoder level <paramref name="level"/> downsamples time: the last <c>log2(SpatialCompression/TemporalCompression)</c>
    /// spatial levels. MovieRoll's rule, not the 13B <see cref="HunyuanVideoVaeConfig.StageHasTemporalResample"/> rule.</summary>
    public bool EncoderTemporalDownsample(int level) =>
        EncoderSpatialDownsample(level) && level >= Log2(SpatialCompression / TemporalCompression);

    /// <summary>Decoder level <paramref name="level"/> (reversed order) upsamples space: the first <c>log2(SpatialCompression)</c> levels.</summary>
    public bool DecoderSpatialUpsample(int level) => level < Log2(SpatialCompression);

    /// <summary>Decoder level <paramref name="level"/> (reversed order) upsamples time: the first <c>log2(TemporalCompression)</c> levels.</summary>
    public bool DecoderTemporalUpsample(int level) => level < Log2(TemporalCompression);

    /// <summary>Checks that the layers can be built from this configuration: positive channel counts, power-of-two spatial and temporal
    /// factors, a spatial factor equal to 2^(levels−1), a temporal factor that divides the spatial one, and first and last channel counts
    /// divisible by the latent width. Shuffle-group divisibility is checked where each resampler is built.</summary>
    public void Validate()
    {
        if (InChannels <= 0 || OutChannels <= 0 || LatentChannels <= 0 || LayersPerBlock <= 0)
            throw new ArgumentException("Channel counts and LayersPerBlock must be positive.");
        if (BlockOutChannels.Length == 0) throw new ArgumentException("BlockOutChannels must not be empty.");
        if (!IsPowerOfTwo(SpatialCompression)) throw new ArgumentException($"SpatialCompression {SpatialCompression} must be a power of two.");
        if (!IsPowerOfTwo(TemporalCompression)) throw new ArgumentException($"TemporalCompression {TemporalCompression} must be a power of two.");
        if (SpatialCompression != 1 << (BlockOutChannels.Length - 1))
            throw new ArgumentException($"SpatialCompression {SpatialCompression} must equal 2^(levels-1) = {1 << (BlockOutChannels.Length - 1)}.");
        if (TemporalCompression > SpatialCompression)
            throw new ArgumentException($"TemporalCompression {TemporalCompression} must not exceed SpatialCompression {SpatialCompression}.");
        if (BlockOutChannels[^1] % (2 * LatentChannels) != 0)
            throw new ArgumentException($"The last block channels {BlockOutChannels[^1]} must be divisible by 2·LatentChannels ({2 * LatentChannels}).");
        if (BlockOutChannels[0] % LatentChannels != 0)
            throw new ArgumentException($"The first block channels {BlockOutChannels[0]} must be divisible by LatentChannels ({LatentChannels}).");
    }

    private static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

    private static int Log2(int value) => (int)Math.Round(Math.Log2(value));
}
