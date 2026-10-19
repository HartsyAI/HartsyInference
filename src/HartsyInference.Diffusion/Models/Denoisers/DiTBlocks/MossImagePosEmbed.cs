namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image coordinate rules: which token sits at which <c>(r, t, h, w)</c> coordinate. These are model-owned; the
/// rotary tables built from them live in <see cref="MossImageRope"/>.</summary>
public static class MossImagePosEmbed
{
    /// <summary>Coordinates for the joint sequence: the centred image grid followed by the text tokens, <c>NumAxes</c> floats each.</summary>
    public static float[] JointCoords(int height, int width, int textLength)
    {
        int imageTokens = height * width;
        float[] coords = new float[(imageTokens + textLength) * MossImageRope.NumAxes];
        float rowCenter = (height - 1) / 2f, colCenter = (width - 1) / 2f;
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                int at = (row * width + col) * MossImageRope.NumAxes;
                coords[at + 1] = textLength;
                coords[at + 2] = row - rowCenter;
                coords[at + 3] = col - colCenter;
            }
        }
        for (int i = 0; i < textLength; i++) coords[(imageTokens + i) * MossImageRope.NumAxes + 1] = i;
        return coords;
    }

    /// <summary>Coordinates for the text refiner: text token <c>i</c> sits at <c>(0, i, 0, 0)</c>.</summary>
    public static float[] TextCoords(int textLength)
    {
        float[] coords = new float[textLength * MossImageRope.NumAxes];
        for (int i = 0; i < textLength; i++) coords[i * MossImageRope.NumAxes + 1] = i;
        return coords;
    }
}
