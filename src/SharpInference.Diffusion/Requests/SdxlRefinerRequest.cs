namespace SharpInference.Diffusion.Requests;

/// <summary>Request parameters for SDXL refiner generation. The refiner runs as an img2img pass with a partially-noised source latent and the SDXL refiner UNet's aesthetic-score-based ADM conditioning.</summary>
/// <remarks>
/// The refiner UNet expects a 5-scalar ADM vector (orig_h, orig_w, crop_top, crop_left, aesthetic_score),
/// which differs from the SDXL base's 6-scalar (target_h/w replaces aesthetic_score). Both <see cref="AestheticScore"/>
/// and <see cref="NegativeAestheticScore"/> are sinusoidally encoded and concatenated with the spatial scalars.
/// </remarks>
public record SdxlRefinerRequest : ImageToImageRequest
{
    /// <summary>Positive aesthetic score (higher = stronger refinement toward "aesthetic" output). Default 6.0 matches diffusers' StableDiffusionXLImg2ImgPipeline default.</summary>
    public float AestheticScore { get; init; } = 6.0f;

    /// <summary>Negative aesthetic score, used for the unconditional CFG branch. Default 2.5 matches diffusers.</summary>
    public float NegativeAestheticScore { get; init; } = 2.5f;
}
