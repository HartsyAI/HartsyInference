namespace HartsyInference.Engine.Requests;

/// <summary>Inpaint mask configuration: the mask image plus grow/blur/shrink adjustments applied to it.</summary>
public sealed record Inpaint
{
    /// <summary>The mask image (white = regenerate, black = keep).</summary>
    public required ImageData Mask { get; init; }

    /// <summary>Mask Grow in SwarmUI units: the mask expands by (Grow + 1) / 2 px per side.</summary>
    public int Grow { get; init; }

    /// <summary>Gaussian blur radius applied to the mask edge, in pixels.</summary>
    public int Blur { get; init; }

    /// <summary>"Inpaint only masked" crop padding in pixels around the mask bounds; nonzero enables the crop.</summary>
    public int ShrinkGrow { get; init; }

    /// <summary>Enables the crop even at <see cref="ShrinkGrow"/> 0, which crops exactly to the mask bounds.</summary>
    public bool CropToMask { get; init; }

    /// <summary>True when the request asks for an "inpaint only masked" crop.</summary>
    public bool CropsToMask => CropToMask || ShrinkGrow != 0;
}
