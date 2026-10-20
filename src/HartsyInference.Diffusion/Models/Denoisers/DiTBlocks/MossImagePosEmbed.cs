namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image coordinate rules: which token sits at which <c>(r, t, h, w)</c> coordinate. These are model-owned; the
/// rotary tables built from them live in <see cref="MossImageRope"/>.</summary>
public static class MossImagePosEmbed
{
    /// <summary>Coordinates for the joint sequence: the centred image grid followed by the text tokens, <c>NumAxes</c> floats each.</summary>
    public static float[] JointCoords(int height, int width, int textLength) => JointCoords(height, width, textLength, []);

    /// <summary>Coordinates for the joint sequence with reference images appended after the text (the Ostris edit layout):
    /// reference <c>i</c> sits on its own <paramref name="referenceGrids"/> grid at <c>(i + 1, textLength, centred row,
    /// centred column)</c>.</summary>
    public static float[] JointCoords(int height, int width, int textLength, IReadOnlyList<(int Height, int Width)> referenceGrids)
    {
        ArgumentNullException.ThrowIfNull(referenceGrids);
        int imageTokens = height * width;
        float[] coords = new float[(imageTokens + textLength + CountTokens(referenceGrids)) * MossImageRope.NumAxes];
        FillGrid(coords, 0, 0, textLength, height, width);
        for (int i = 0; i < textLength; i++) coords[(imageTokens + i) * MossImageRope.NumAxes + 1] = i;
        FillReferences(coords, imageTokens + textLength, textLength, referenceGrids);
        return coords;
    }

    /// <summary>Coordinates for reference images alone, as a prepass sees them: the reference rows of
    /// <see cref="JointCoords(int, int, int, IReadOnlyList{ValueTuple{int, int}})"/>.</summary>
    public static float[] ReferenceCoords(IReadOnlyList<(int Height, int Width)> referenceGrids, int textLength)
    {
        ArgumentNullException.ThrowIfNull(referenceGrids);
        float[] coords = new float[CountTokens(referenceGrids) * MossImageRope.NumAxes];
        FillReferences(coords, 0, textLength, referenceGrids);
        return coords;
    }

    /// <summary>Coordinates for the text refiner: text token <c>i</c> sits at <c>(0, i, 0, 0)</c>.</summary>
    public static float[] TextCoords(int textLength)
    {
        float[] coords = new float[textLength * MossImageRope.NumAxes];
        for (int i = 0; i < textLength; i++) coords[i * MossImageRope.NumAxes + 1] = i;
        return coords;
    }

    private static int CountTokens(IReadOnlyList<(int Height, int Width)> grids)
    {
        int tokens = 0;
        foreach ((int h, int w) in grids)
        {
            if (h < 1 || w < 1)
                throw new ArgumentOutOfRangeException(nameof(grids), $"A reference grid must be positive, got {h}x{w}.");
            tokens += h * w;
        }
        return tokens;
    }

    private static void FillReferences(float[] coords, int firstToken, int textLength, IReadOnlyList<(int Height, int Width)> grids)
    {
        int token = firstToken;
        for (int i = 0; i < grids.Count; i++)
        {
            FillGrid(coords, token, i + 1, textLength, grids[i].Height, grids[i].Width);
            token += grids[i].Height * grids[i].Width;
        }
    }

    /// <summary>Writes a raster grid's <c>(index, frame, centred row, centred column)</c> from token <paramref name="firstToken"/>.</summary>
    private static void FillGrid(float[] coords, int firstToken, int index, int frame, int height, int width)
    {
        float rowCenter = (height - 1) / 2f, colCenter = (width - 1) / 2f;
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                int at = (firstToken + row * width + col) * MossImageRope.NumAxes;
                coords[at] = index;
                coords[at + 1] = frame;
                coords[at + 2] = row - rowCenter;
                coords[at + 3] = col - colCenter;
            }
        }
    }
}
