namespace SharpInference.Diffusion.Utilities;

/// <summary>Latent-space architecture identifier used by preview encoders
/// (latent2rgb / TAESD) to pick the right factor matrix or weight set.
/// The values track each model family's VAE channel count and scale conventions.
/// <see cref="Unknown"/> means the producer didn't tag the latent — preview
/// encoders should skip rather than guess.</summary>
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
    /// <summary>Flux.2 — 16-channel latent, distinct factors from Flux.1.
    /// Falls back to Flux.1 factors if Flux.2-specific ones aren't published.</summary>
    Flux2 = 5,
    /// <summary>Chroma — reuses the Flux VAE, so Flux factors apply.</summary>
    Chroma = 6,
    /// <summary>AuraFlow — 4-channel latent, SDXL-like VAE.</summary>
    AuraFlow = 7,
    /// <summary>F-Lite — 16-channel latent (SD3-family VAE).</summary>
    FLite = 8,
    /// <summary>Z-Image — reuses the Flux VAE, so Flux factors apply.</summary>
    ZImage = 9,
    /// <summary>Anima (Cosmos-Predict2 2B) — 16-channel Qwen-Image VAE, 8× downscale. Same channel count as Flux/SD3
    /// so Flux factors are a reasonable preview approximation until Qwen-Image-specific factors are derived.</summary>
    Anima = 10,
}
