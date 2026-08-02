using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Config for the SeedVR2 causal video VAE (<c>s8_c16_t4_inflation_sd3</c>): SD-inflated 3D,
/// channels [128,256,512,512], 2 encoder resnets / 3 decoder resnets per stage, GroupNorm(32, 1e-6),
/// per-frame norm statistics, no quant convs (encoder conv_out emits mean‖logvar directly). 8× spatial via
/// downsamplers on stages 0–2; 4× temporal via stages 1–2 (stage-0 downsampler is spatial-only, kernel
/// (1,3,3)). Decoder mirrors: upsamplers 0–1 are ×8 spatiotemporal shuffles, upsampler 2 is ×4 spatial.</summary>
public sealed record SeedVr2VaeConfig
{
    /// <summary>Per-stage channel widths, encoder order.</summary>
    public int[] BlockOutChannels { get; init; } = [128, 256, 512, 512];

    /// <summary>Latent channels (16); encoder conv_out emits 2× this (mean‖logvar).</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Resnets per encoder stage (decoder stages carry one more).</summary>
    public int LayersPerBlock { get; init; } = 2;

    /// <summary>GroupNorm groups.</summary>
    public int NormGroups { get; init; } = 32;

    /// <summary>GroupNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>Latent scaling: DiT-space latent = (raw − shift) × scale.</summary>
    public float ScalingFactor { get; init; } = 0.9152f;

    /// <summary>Activation/conv-weight dtype inside the VAE. BF16 halves the fp32 activation peak that OOMs
    /// 24 GB at 720p-area (the reference runs bf16); pixel/latent boundaries stay F32 either way. CUDA-only —
    /// the CPU/Vulkan conv path and the host-parity tests keep F32.</summary>
    public DType ActivationDType { get; init; } = DType.F32;

    /// <summary>Pixels per latent cell on H/W.</summary>
    public int SpatialCompression { get; init; } = 8;

    /// <summary>Frames per latent frame: T_lat = (F−1)/4 + 1.</summary>
    public int TemporalCompression { get; init; } = 4;

    /// <summary>Whether encoder stage <paramref name="stage"/> has a downsampler (stages 0–2).</summary>
    public bool StageHasDownsample(int stage) => stage < BlockOutChannels.Length - 1;

    /// <summary>Whether that downsampler also halves time (stages 1–2; checkpoint kernel (3,3,3) vs (1,3,3)).</summary>
    public bool StageDownsamplesTime(int stage) => stage is 1 or 2;

    /// <summary>Whether decoder stage <paramref name="stage"/> has an upsampler (stages 0–2).</summary>
    public bool StageHasUpsample(int stage) => stage < BlockOutChannels.Length - 1;

    /// <summary>Whether that upsampler also doubles time (stages 0–1; upscale_conv ratio ×8 vs ×4).</summary>
    public bool StageUpsamplesTime(int stage) => stage is 0 or 1;

    /// <summary>Published SeedVR2 VAE configuration.</summary>
    public static SeedVr2VaeConfig Default => new();
}
