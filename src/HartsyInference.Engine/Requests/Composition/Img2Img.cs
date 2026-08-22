namespace HartsyInference.Engine.Requests;

/// <summary>Image-to-image init: the starting image plus how much creative freedom (denoise strength) to allow.</summary>
public sealed record Img2Img
{
    /// <summary>The init image to diffuse from.</summary>
    public required ImageData InitImage { get; init; }

    /// <summary>Creativity / denoise strength in 0..1 (0 keeps the init, 1 ignores it). Ignored under <see cref="Img2ImgMode.Reference"/>, which conditions on the init image at full strength.</summary>
    public double Creativity { get; init; } = 0.6;

    /// <summary>How the family should consume <see cref="InitImage"/>; <see cref="Img2ImgMode.Auto"/> lets the resolved family decide, which is unambiguous for every family except Qwen-Image (the only one offering both).</summary>
    public Img2ImgMode Mode { get; init; } = Img2ImgMode.Auto;
}
