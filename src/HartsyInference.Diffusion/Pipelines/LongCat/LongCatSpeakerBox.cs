namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>A region of the source portrait in its own pixels, the reference's <c>[y_min, x_min, y_max, x_max]</c>: rows <c>YMin</c> up to
/// <c>YMax</c> and columns <c>XMin</c> up to <c>XMax</c>, the ends excluded. A box past the image edge is clipped to it.</summary>
public readonly record struct LongCatSpeakerBox(int YMin, int XMin, int YMax, int XMax)
{
    /// <summary>True when the box has non-negative corners and at least one pixel.</summary>
    public bool IsValid => YMin >= 0 && XMin >= 0 && YMax > YMin && XMax > XMin;

    internal bool Contains(int y, int x) => y >= YMin && y < YMax && x >= XMin && x < XMax;
}
