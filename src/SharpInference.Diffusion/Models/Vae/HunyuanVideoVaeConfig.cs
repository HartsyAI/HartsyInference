namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Configuration for the HunyuanVideo 3D causal VAE (<c>AutoencoderKLHunyuanVideo</c>), used by
/// HunyuanVideo and Kandinsky 5.0 T2V. Defaults match <c>vae/config.json</c> of both repos: 16-channel
/// latent, 8× spatial / 4× temporal compression, block channels [128, 256, 512, 512], GroupNorm(32),
/// replicate-padded causal convs, mid-block single-head attention with a frame-causal mask.</summary>
public sealed record HunyuanVideoVaeConfig
{
    /// <summary>RGB input/output channels.</summary>
    public int InChannels { get; init; } = 3;

    /// <summary>Latent channels (encoder emits 2× for mean+logvar before <c>quant_conv</c>).</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Per-stage channel counts, encoder order (decoder mirrors them reversed).</summary>
    public int[] BlockOutChannels { get; init; } = [128, 256, 512, 512];

    /// <summary>Resnets per encoder down block (decoder up blocks use one more).</summary>
    public int LayersPerBlock { get; init; } = 2;

    /// <summary>GroupNorm group count.</summary>
    public int NormGroups { get; init; } = 32;

    /// <summary>GroupNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>Latent scaling factor — model-space latent = raw VAE latent × this. HunyuanVideo: 0.476986.</summary>
    public float ScalingFactor { get; init; } = 0.476986f;

    /// <summary>Spatial down/upsample ratio (8 → 3 spatial resample stages).</summary>
    public int SpatialCompression { get; init; } = 8;

    /// <summary>Temporal down/upsample ratio (4 → 2 temporal resample stages). <c>T_lat = (F−1)/4 + 1</c>.</summary>
    public int TemporalCompression { get; init; } = 4;

    /// <summary>Whether the mid block carries the single attention layer.</summary>
    public bool MidBlockAttention { get; init; } = true;

    /// <summary>Stock HunyuanVideo / Kandinsky 5 T2V configuration.</summary>
    public static HunyuanVideoVaeConfig Default => new();

    /// <summary>Whether stage <paramref name="i"/> (of <see cref="BlockOutChannels"/>.Length stages) resamples spatially — per the diffusers rule: the first <c>log2(spatial)</c> stages.</summary>
    public bool StageHasSpatialResample(int i) => i < (int)Math.Log2(SpatialCompression);

    /// <summary>Whether stage <paramref name="i"/> resamples temporally — per the diffusers rule for temporal_compression=4: the last <c>log2(temporal)</c> non-final stages.</summary>
    public bool StageHasTemporalResample(int i)
    {
        int numBlocks = BlockOutChannels.Length;
        int numTime = (int)Math.Log2(TemporalCompression);
        return i >= numBlocks - 1 - numTime && i != numBlocks - 1;
    }
}
