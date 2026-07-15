namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS image-to-3D configuration (the <c>microsoft/TRELLIS-image-large</c> checkpoint): a two-stage
/// flow-matching system — a dense sparse-structure stage (16³ latent → 64³ occupancy → active-voxel coords) and a
/// sparse structured-latent (SLAT) stage over those voxels → Gaussian-splat / mesh / radiance-field decoders.
/// Exact values come from the ckpt configs (<c>ss_flow_img_dit_L_16l8</c>, <c>slat_flow_img_dit_L_64l8p2</c>,
/// <c>ss_vae_conv3d_16l8</c>, <c>slat_vae_*_swin8_B_64l8</c>); the build plan is <c>docs/Checklists/TRELLIS_BUILD_PLAN.md</c>.
/// <b>Validation-gated</b> — every sub-model is parity-verified against the reference before it counts as done.</summary>
public sealed record TrellisConfig
{
    #region Conditioner (DINOv2 ViT-L/14 with registers)
    /// <summary>DINOv2 conditioner hidden dim (<c>dinov2_vitl14_reg</c> = 1024); patchtokens are the pre-norm
    /// features passed through a final <c>layer_norm</c>, then used as cross-attention condition.</summary>
    public int CondDim { get; init; } = 1024;

    /// <summary>Conditioner input resolution (square); TRELLIS feeds a premultiplied 518² RGB.</summary>
    public int CondImageSize { get; init; } = 518;
    #endregion

    #region Stage 1 — sparse-structure flow (dense 16³ DiT)
    /// <summary>SS-flow latent grid resolution (16³).</summary>
    public int SsResolution { get; init; } = 16;

    /// <summary>SS-flow in/out channels (8) and patch size (1).</summary>
    public int SsChannels { get; init; } = 8;
    public int SsPatchSize { get; init; } = 1;

    /// <summary>SS-flow DiT width / blocks / heads / mlp-ratio (DiT-L: 1024 / 24 / 16 / 4).</summary>
    public int SsModelChannels { get; init; } = 1024;
    public int SsNumBlocks { get; init; } = 24;
    public int SsNumHeads { get; init; } = 16;
    public int SsMlpRatio { get; init; } = 4;
    #endregion

    #region Stage 1 — sparse-structure VAE decoder (conv3d, 16³ latent → 64³ occupancy)
    /// <summary>SS-VAE decoder latent channels (8) → occupancy out channels (1); channel schedule [512,128,32],
    /// 2 res-blocks + 2 middle, ×2 upsample per stage → 64³.</summary>
    public int SsVaeLatentChannels { get; init; } = 8;
    public int SsVaeOutChannels { get; init; } = 1;
    public int[] SsVaeDecoderChannels { get; init; } = [512, 128, 32];
    public int SsVaeNumResBlocks { get; init; } = 2;
    public int SsVaeNumResBlocksMiddle { get; init; } = 2;
    public int SsVaeOutResolution { get; init; } = 64;
    #endregion

    #region Stage 2 — structured-latent (SLAT) flow (sparse DiT over active voxels @ 64³)
    /// <summary>SLAT-flow resolution (64), in/out channels (8), patch size (2), io res-blocks (2) with
    /// io_block_channels [128]; DiT-L width/blocks/heads (1024 / 24 / 16), APE pos, qk_rms_norm.</summary>
    public int SlatResolution { get; init; } = 64;
    public int SlatChannels { get; init; } = 8;
    public int SlatPatchSize { get; init; } = 2;
    public int SlatNumIoResBlocks { get; init; } = 2;
    public int[] SlatIoBlockChannels { get; init; } = [128];
    public int SlatModelChannels { get; init; } = 1024;
    public int SlatNumBlocks { get; init; } = 24;
    public int SlatNumHeads { get; init; } = 16;
    public int SlatMlpRatio { get; init; } = 4;
    #endregion

    #region Stage 2 — SLAT VAE decoders (sparse, swin window-8, DiT-B)
    /// <summary>SLAT-decoder width / blocks / heads (DiT-B: 768 / 12 / 12), swin window size 8.</summary>
    public int SlatDecModelChannels { get; init; } = 768;
    public int SlatDecNumBlocks { get; init; } = 12;
    public int SlatDecNumHeads { get; init; } = 12;
    public int SlatDecWindowSize { get; init; } = 8;

    /// <summary>Gaussian-splat decoder: gaussians per active voxel (32), voxel size, softplus scaling bias/opacity
    /// bias per the ckpt <c>representation_config</c>.</summary>
    public int GsNumGaussiansPerVoxel { get; init; } = 32;
    public float GsVoxelSize { get; init; } = 1.5f;
    public float GsScalingBias { get; init; } = 0.004f;
    public float GsOpacityBias { get; init; } = 0.1f;
    #endregion

    /// <summary>The shipped <c>microsoft/TRELLIS-image-large</c> configuration (all defaults).</summary>
    public static TrellisConfig ImageLarge => new();
}
