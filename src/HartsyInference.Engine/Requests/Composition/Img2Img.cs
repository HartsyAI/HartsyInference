namespace HartsyInference.Engine.Requests;

/// <summary>Image-to-image init: the starting image plus how much creative freedom (denoise strength) to allow.</summary>
public sealed record Img2Img
{
    /// <summary>The init image to diffuse from.</summary>
    public required ImageData InitImage { get; init; }

    /// <summary>Creativity / denoise strength in 0..1 (0 keeps the init, 1 ignores it).</summary>
    public double Creativity { get; init; } = 0.6;
}
