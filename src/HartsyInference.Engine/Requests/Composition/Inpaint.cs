namespace HartsyInference.Engine.Requests;

/// <summary>Inpaint mask configuration: the mask image plus grow/blur/shrink adjustments applied to it.</summary>
public sealed record Inpaint
{
    /// <summary>The mask image (white = regenerate, black = keep).</summary>
    public required ImageData Mask { get; init; }

    /// <summary>Pixels to grow the mask outward before use.</summary>
    public int Grow { get; init; }

    /// <summary>Gaussian blur radius applied to the mask edge, in pixels.</summary>
    public int Blur { get; init; }

    /// <summary>Signed shrink/grow adjustment applied after <see cref="Grow"/> (negative shrinks).</summary>
    public int ShrinkGrow { get; init; }
}
