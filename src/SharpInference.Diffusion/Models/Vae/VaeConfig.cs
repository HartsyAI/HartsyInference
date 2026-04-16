namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Configuration for AutoencoderKL VAE. Covers SD1.5, SDXL, SD3, and Flux variants.</summary>
public record VaeConfig
{
    /// <summary>Number of latent channels (4 for SD1.5/SDXL, 16 for SD3/Flux).</summary>
    public int LatentChannels { get; init; } = 4;

    /// <summary>Output channel counts per block stage: [128, 256, 512, 512] for all standard configs.</summary>
    public int[] BlockOutChannels { get; init; } = [128, 256, 512, 512];

    /// <summary>Number of ResNet layers per encoder down-block. Decoder uses LayersPerBlock + 1.</summary>
    public int LayersPerBlock { get; init; } = 2;

    /// <summary>Number of groups for GroupNorm (always 32 in standard configs).</summary>
    public int NormNumGroups { get; init; } = 32;

    /// <summary>Scaling factor applied to latents. Model-specific.</summary>
    public float ScalingFactor { get; init; } = 0.18215f;

    /// <summary>Shift factor for SD3/Flux latent centering. Null for SD1.5/SDXL.</summary>
    public float? ShiftFactor { get; init; }

    /// <summary>Whether this model uses a 1x1 post_quant_conv before the decoder (SD1.5/SDXL: true, SD3/Flux: false).</summary>
    public bool UsePostQuantConv { get; init; } = true;

    /// <summary>Whether this model uses a 1x1 quant_conv after the encoder (SD1.5/SDXL: true, SD3/Flux: false).</summary>
    public bool UseQuantConv { get; init; } = true;

    /// <summary>Sample size for tiled decode (512 for SD1.5, 1024 for SDXL/SD3/Flux).</summary>
    public int SampleSize { get; init; } = 512;

    /// <summary>GroupNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>Preset for Stable Diffusion 1.5.</summary>
    public static VaeConfig Sd15 => new()
    {
        LatentChannels = 4,
        ScalingFactor = 0.18215f,
        ShiftFactor = null,
        UsePostQuantConv = true,
        UseQuantConv = true,
        SampleSize = 512,
    };

    /// <summary>Preset for SDXL.</summary>
    public static VaeConfig Sdxl => new()
    {
        LatentChannels = 4,
        ScalingFactor = 0.13025f,
        ShiftFactor = null,
        UsePostQuantConv = true,
        UseQuantConv = true,
        SampleSize = 1024,
    };

    /// <summary>Preset for Stable Diffusion 3.</summary>
    public static VaeConfig Sd3 => new()
    {
        LatentChannels = 16,
        ScalingFactor = 1.5305f,
        ShiftFactor = 0.0609f,
        UsePostQuantConv = false,
        UseQuantConv = false,
        SampleSize = 1024,
    };

    /// <summary>Preset for Flux.1.</summary>
    public static VaeConfig Flux => new()
    {
        LatentChannels = 16,
        ScalingFactor = 0.3611f,
        ShiftFactor = 0.1159f,
        UsePostQuantConv = false,
        UseQuantConv = false,
        SampleSize = 1024,
    };
}
