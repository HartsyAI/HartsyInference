namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Position-id builder for Hunyuan Image's packed-grid RoPE. Tokens are laid out row-major over the packed dims
/// <c>(h, w)</c> for image or <c>(t, h, w)</c> for video, and token <c>s</c> sits at its row-major coordinates on each axis.
/// Rows are token-major with one value per axis, the layout <see cref="Core.Rope.MultiAxisRopeTables"/> reads.</summary>
public static class HunyuanImagePosEmbed
{
    /// <summary>Flat row count of a packed grid: the product of its dims.</summary>
    public static int TokenCount(ReadOnlySpan<int> packedDims)
    {
        int seqLen = 1;
        for (int i = 0; i < packedDims.Length; i++)
            seqLen = checked(seqLen * packedDims[i]);
        return seqLen;
    }

    /// <summary>Rows for every token of the grid, one value per axis.</summary>
    public static float[] GridRows(ReadOnlySpan<int> packedDims)
    {
        int seqLen = TokenCount(packedDims);
        int axes = packedDims.Length;
        float[] rows = new float[checked(seqLen * axes)];
        for (int s = 0; s < seqLen; s++)
        {
            int rem = s;
            for (int axis = axes - 1; axis >= 0; axis--)
            {
                rows[s * axes + axis] = rem % packedDims[axis];
                rem /= packedDims[axis];
            }
        }
        return rows;
    }
}
