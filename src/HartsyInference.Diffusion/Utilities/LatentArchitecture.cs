namespace HartsyInference.Diffusion.Utilities;

/// <summary>Latent-space architecture identifier used by preview encoders (latent2rgb / TAESD) to pick the right factor matrix or weight set, tracking each model family's VAE channel count and scale conventions; <see cref="Unknown"/> means the producer didn't tag the latent, so preview encoders should skip rather than guess.</summary>
public enum LatentArchitecture
{
    Unknown = 0,
    /// <summary>SD 1.5 — 4-channel latent, 8x downscale vs pixel space.</summary>
    Sd15 = 1,
    /// <summary>SDXL — 4-channel latent, 8x downscale (different factors from SD1.5).</summary>
    Sdxl = 2,
    /// <summary>SD3 / SD3.5 — 16-channel latent, 8x downscale.</summary>
    Sd3 = 3,
    /// <summary>Flux.1 (dev / schnell) — 16-channel latent, 8x downscale.</summary>
    Flux = 4,
    /// <summary>Flux.2 — canonical 32-channel latent after its 128-channel patch representation is unpatchified.</summary>
    Flux2 = 5,
    /// <summary>Chroma — reuses the Flux VAE, so Flux factors apply.</summary>
    Chroma = 6,
    /// <summary>AuraFlow — 4-channel latent, SDXL-like VAE.</summary>
    AuraFlow = 7,
    /// <summary>F-Lite — 16-channel latent (SD3-family VAE).</summary>
    FLite = 8,
    /// <summary>Z-Image — reuses the Flux VAE, so Flux factors apply.</summary>
    ZImage = 9,
    /// <summary>Anima (Cosmos-Predict2 2B) — 16-channel Qwen-Image VAE, 8× downscale; same channel count as Flux/SD3 so Flux factors are a reasonable preview approximation until Qwen-Image-specific factors are derived.</summary>
    Anima = 10,
    /// <summary>Wan video — 16-channel Wan 2.1 or 48-channel Wan 2.2 3-D latent.</summary>
    Wan = 11,
    /// <summary>LTX-Video — 128-channel 3-D latent, 32× spatial downscale.</summary>
    Ltx = 12,
    /// <summary>Chroma Radiance — pixel-space (no VAE); the "latent" IS the RGB image in [-1, 1], so previews convert it directly without a factor matrix.</summary>
    ChromaRadiance = 13,
    /// <summary>Zeta-Chroma — pixel-space Z-Image S3-DiT (no VAE). Direct RGB preview like <see cref="ChromaRadiance"/>.</summary>
    ZetaChroma = 14,
    /// <summary>HunyuanVideo and Kandinsky 5 Video — 16-channel 3-D latent.</summary>
    HunyuanVideo = 15,
    /// <summary>MiniMax H3 video — 24-channel 3-D latent.</summary>
    MiniMaxH3 = 16,
    /// <summary>Hunyuan Image 2.1 — 64-channel image latent.</summary>
    HunyuanImage = 17,
    /// <summary>Mage-Flow — 128-channel image latent. Uses an approximate deterministic projection until
    /// calibrated latent-to-RGB factors are published.</summary>
    MageFlow = 18,
}
