namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Configuration for AutoencoderKL VAE. Covers SD1.5, SDXL, SD3, and Flux variants.</summary>
public sealed record VaeConfig
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

    /// <summary>Preset for Flux.2 (32-channel VAE, 8× spatial downscale, post_quant_conv used). The
    /// effective 16× downscale for the transformer comes from a 2×2 patchify on top of the latent
    /// (handled in the pipeline). Latent normalization is done via BatchNorm-like statistics
    /// (<c>bn.running_mean/var</c>) — applied at pipeline boundary, not inside the VAE itself.
    /// Verified against Comfy-Org/flux2-klein/flux2-vae.safetensors.</summary>
    public static VaeConfig Flux2 => new()
    {
        LatentChannels = 32,
        ScalingFactor = 1.0f,           // unused — Flux.2 uses BN-style normalization, applied by pipeline
        ShiftFactor = null,             // ditto
        UsePostQuantConv = true,        // post_quant_conv: [32, 32, 1, 1]
        UseQuantConv = true,            // quant_conv: [64, 64, 1, 1]
        SampleSize = 1024,
    };

    /// <summary>Preset for Chroma (reuses Flux.1 VAE).</summary>
    public static VaeConfig Chroma => Flux;

    /// <summary>Preset for Z-Image (reuses Flux.1 VAE verbatim — same scale=0.3611, shift=0.1159, 16 channels, 8× downscale).</summary>
    public static VaeConfig ZImage => Flux;

    /// <summary>Preset for AuraFlow (reuses SDXL-compatible VAE with 4-channel latent).</summary>
    public static VaeConfig AuraFlow => Sdxl;

    /// <summary>Preset for Hunyuan Image 2.1 (64-channel latent, 32× downscale, 6 up levels). Derived from the released VAE weights + ComfyUI <c>latent_formats.HunyuanImage21</c> (scale 0.75289); block channels match diffusers' conversion script.</summary>
    public static VaeConfig HunyuanImage => new()
    {
        LatentChannels = 64,
        BlockOutChannels = [128, 256, 512, 512, 1024, 1024],
        ScalingFactor = 0.75289f,
        ShiftFactor = null,
        UsePostQuantConv = false,
        UseQuantConv = false,
        SampleSize = 2048,
    };

    /// <summary>Per-channel mean values for VAE latent post-denoising rescale. When set, the decode path
    /// does <c>latent_rescaled[b, c, h, w] = latent[b, c, h, w] * LatentsStd[c] + LatentsMean[c]</c>
    /// (per-channel) instead of the scalar <c>latent / ScalingFactor + ShiftFactor</c>. Used by the
    /// Qwen-Image and Wan VAE families. Length must equal <see cref="LatentChannels"/> when set.</summary>
    public float[]? LatentsMean { get; init; }

    /// <summary>Per-channel std values, paired with <see cref="LatentsMean"/>. See remarks above.</summary>
    public float[]? LatentsStd { get; init; }

    /// <summary>Preset for Qwen-Image (3D causal autoencoder, WAN 2.1 family). Used by Anima and
    /// Qwen-Image proper. Uses per-channel <c>latents_mean</c> / <c>latents_std</c> rescale before
    /// decode (matches <c>AutoencoderKLQwenImage.config</c> in diffusers, applied by the pipeline as
    /// <c>latents = latents * std + mean</c> per channel BEFORE <c>vae.decode</c>).</summary>
    public static VaeConfig QwenImage => new()
    {
        LatentChannels = 16,
        // Scalar fallbacks (unused when LatentsMean/Std are set, but kept for API compatibility).
        ScalingFactor = 1.0f,
        ShiftFactor = 0.0f,
        UsePostQuantConv = false,
        UseQuantConv = false,
        SampleSize = 1024,
        // Per-channel statistics from diffusers' AutoencoderKLQwenImage.config (16 channels each).
        LatentsMean = [
            -0.7571f, -0.7089f, -0.9113f,  0.1075f, -0.1745f,  0.9653f, -0.1517f,  1.5508f,
             0.4134f, -0.0715f,  0.5517f, -0.3632f, -0.1922f, -0.9497f,  0.2503f, -0.2921f,
        ],
        LatentsStd = [
            2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
            3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f,
        ],
    };
}
