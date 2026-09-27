namespace HartsyInference.Engine.Requests;

/// <summary>How a family should consume <see cref="Img2Img.InitImage"/>. Most families offer exactly one of these, but a Qwen-Image-Edit build offers both, so the choice cannot be left implicit — a caller who wanted an edit and silently got a strength-based denoise (or the reverse) gets a plausible image that answers the wrong question.</summary>
public enum Img2ImgMode
{
    /// <summary>Pick whichever mode the resolved variant declares; where it declares both, an edit build (Qwen-Image-Edit) prefers <see cref="Reference"/>, which is what its init image is for, unless an inpaint mask asks for <see cref="Denoise"/>.</summary>
    Auto,

    /// <summary>Classic image-to-image: encode the init image, add noise at the step selected by <see cref="Img2Img.Creativity"/>, and denoise from there.</summary>
    Denoise,

    /// <summary>Reference-image editing: encode the init image into in-context reference latents that condition every step at full strength. <see cref="Img2Img.Creativity"/> does not apply.</summary>
    Reference,
}
