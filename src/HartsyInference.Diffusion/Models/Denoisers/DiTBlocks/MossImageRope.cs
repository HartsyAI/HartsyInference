using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image's four-axis rotate-half RoPE. Every token has a coordinate <c>(r, t, h, w)</c>: reference index,
/// sequence position, row and column. Each axis owns <c>headDim/8</c> frequency bands
/// <c>1/base^(j/bands)</c>, so the four axes fill half the head, and channel <c>i</c> pairs with <c>i + headDim/2</c>.
/// Image tokens come first at <c>(0, textLength, row − (H−1)/2, col − (W−1)/2)</c>; text token <c>i</c> sits at
/// <c>(0, i, 0, 0)</c>. Reference image <c>i</c> of an edit takes <c>r = i + 1</c>; with none, <c>r</c> is 0 everywhere and
/// its quarter of the head is unrotated.
/// Tables are full-width <c>[1, S, headDim]</c> (both halves repeat the per-pair value), the layout
/// <see cref="Core.Backends.IBackend.ApplyRope"/> takes. Phases are F32 products, as the reference builds them.</summary>
public static class MossImageRope
{
    /// <summary>Coordinate axes per token.</summary>
    public const int NumAxes = 4;

    /// <summary>The per-axis band frequencies, built in F32 as the reference does.</summary>
    public static float[] Bands(int headDim, float ropeBase)
    {
        if (headDim <= 0 || headDim % 8 != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "MOSS-Image RoPE needs a head width that is a multiple of 8.");
        int count = headDim / 8;
        float[] bands = new float[count];
        for (int j = 0; j < count; j++) bands[j] = 1f / MathF.Pow(ropeBase, (float)j / count);
        return bands;
    }

    /// <summary>Tables for the joint sequence: the <paramref name="height"/>×<paramref name="width"/> image grid followed
    /// by <paramref name="textLength"/> text tokens.</summary>
    public static (Tensor Cos, Tensor Sin) Joint(int height, int width, int textLength, int headDim, float ropeBase) =>
        Joint(height, width, textLength, [], headDim, ropeBase);

    /// <summary>Tables for the joint sequence with reference images appended after the text, each reference on its own
    /// <paramref name="referenceGrids"/> grid (the Ostris edit layout, <see cref="References"/>).</summary>
    public static (Tensor Cos, Tensor Sin) Joint(int height, int width, int textLength,
        IReadOnlyList<(int Height, int Width)> referenceGrids, int headDim, float ropeBase)
    {
        ArgumentNullException.ThrowIfNull(referenceGrids);
        int imageTokens = height * width;
        float[] coords = new float[(imageTokens + textLength + CountTokens(referenceGrids)) * NumAxes];
        FillGrid(coords, 0, 0, textLength, height, width);
        for (int i = 0; i < textLength; i++) coords[(imageTokens + i) * NumAxes + 1] = i;
        FillReferences(coords, imageTokens + textLength, textLength, referenceGrids);
        return Tables(coords, headDim, ropeBase);
    }

    /// <summary>Tables for reference images alone, as a prepass sees them: reference <c>i</c> sits at
    /// <c>(i + 1, textLength, row − (h−1)/2, col − (w−1)/2)</c> over its own grid, the Ostris edit coordinates.</summary>
    public static (Tensor Cos, Tensor Sin) References(IReadOnlyList<(int Height, int Width)> referenceGrids, int textLength,
        int headDim, float ropeBase)
    {
        ArgumentNullException.ThrowIfNull(referenceGrids);
        float[] coords = new float[CountTokens(referenceGrids) * NumAxes];
        FillReferences(coords, 0, textLength, referenceGrids);
        return Tables(coords, headDim, ropeBase);
    }

    /// <summary>Tables for the text refiner, which runs on the text tokens alone at their joint positions.</summary>
    public static (Tensor Cos, Tensor Sin) Text(int textLength, int headDim, float ropeBase)
    {
        float[] coords = new float[textLength * NumAxes];
        for (int i = 0; i < textLength; i++) coords[i * NumAxes + 1] = i;
        return Tables(coords, headDim, ropeBase);
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
                int at = (firstToken + row * width + col) * NumAxes;
                coords[at] = index;
                coords[at + 1] = frame;
                coords[at + 2] = row - rowCenter;
                coords[at + 3] = col - colCenter;
            }
        }
    }

    private static unsafe (Tensor Cos, Tensor Sin) Tables(float[] coords, int headDim, float ropeBase)
    {
        float[] bands = Bands(headDim, ropeBase);
        int tokens = coords.Length / NumAxes;
        int half = headDim / 2;
        Tensor cos = new(new TensorShape(1, tokens, headDim), DType.F32);
        Tensor sin = new(new TensorShape(1, tokens, headDim), DType.F32);
        float* c = (float*)cos.DataPointer;
        float* s = (float*)sin.DataPointer;
        for (int t = 0; t < tokens; t++)
        {
            float* cRow = c + (long)t * headDim;
            float* sRow = s + (long)t * headDim;
            for (int i = 0; i < half; i++)
            {
                float phase = coords[t * NumAxes + i / bands.Length] * bands[i % bands.Length];
                float cv = (float)Math.Cos(phase), sv = (float)Math.Sin(phase);
                cRow[i] = cv;
                cRow[i + half] = cv;
                sRow[i] = sv;
                sRow[i + half] = sv;
            }
        }
        return (cos, sin);
    }
}
