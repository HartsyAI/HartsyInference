using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image's four-axis rotate-half RoPE. Every token has a coordinate <c>(r, t, h, w)</c>: reference index,
/// sequence position, row and column. Each axis owns <c>headDim/8</c> frequency bands
/// <c>1/base^(j/bands)</c>, so the four axes fill half the head, and channel <c>i</c> pairs with <c>i + headDim/2</c>.
/// Image tokens come first at <c>(0, textLength, row − (H−1)/2, col − (W−1)/2)</c>; text token <c>i</c> sits at
/// <c>(0, i, 0, 0)</c>. With no reference images <c>r</c> is 0 everywhere, so its quarter of the head is unrotated.
/// Tables are full-width <c>[1, S, headDim]</c> (both halves repeat the per-pair value), the layout
/// <see cref="Core.Backends.IBackend.ApplyRope"/> takes. Phases are F32 products, as the reference builds them.
/// The coordinates come from <see cref="MossImagePosEmbed"/>; <see cref="RopeTableExpander"/> writes the rotate-half rows.</summary>
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
        Tables(MossImagePosEmbed.JointCoords(height, width, textLength), headDim, ropeBase);

    /// <summary>Tables for the text refiner, which runs on the text tokens alone at their joint positions.</summary>
    public static (Tensor Cos, Tensor Sin) Text(int textLength, int headDim, float ropeBase) =>
        Tables(MossImagePosEmbed.TextCoords(textLength), headDim, ropeBase);

    /// <summary>Float phases as the reference builds them: <c>phase = coordinate × band</c> in F32, then cos and sin of the
    /// widened phase, rounded to F32. The pairs are compact per token, one per band in <c>(r, t, h, w)</c> order, and
    /// <see cref="RopeTableExpander"/> duplicates each pair into both halves.</summary>
    private static unsafe (Tensor Cos, Tensor Sin) Tables(float[] coords, int headDim, float ropeBase)
    {
        float[] bands = Bands(headDim, ropeBase);
        int tokens = coords.Length / NumAxes;
        int half = headDim / 2;
        float[] cosPairs = new float[checked(tokens * half)];
        float[] sinPairs = new float[cosPairs.Length];
        for (int t = 0; t < tokens; t++)
        {
            int row = t * half;
            for (int i = 0; i < half; i++)
            {
                float phase = coords[t * NumAxes + i / bands.Length] * bands[i % bands.Length];
                cosPairs[row + i] = (float)Math.Cos(phase);
                sinPairs[row + i] = (float)Math.Sin(phase);
            }
        }

        Tensor cos = new(new TensorShape(1, tokens, headDim), DType.F32);
        Tensor? sin = null;
        bool succeeded = false;
        try
        {
            sin = new Tensor(new TensorShape(1, tokens, headDim), DType.F32);
            int count = checked(tokens * headDim);
            RopeTableExpander.Expand(cosPairs, sinPairs, tokens, headDim, RopeTableLayout.RotateHalf, 0, RopeTablePadding.Identity,
                new Span<float>(cos.DataPointer, count), new Span<float>(sin.DataPointer, count));
            succeeded = true;
            return (cos, sin);
        }
        finally
        {
            if (!succeeded)
            {
                cos.Dispose();
                sin?.Dispose();
            }
        }
    }
}
