using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The fixed sinusoidal position embedding of the upstream <c>AbsolutePositionEmbedder</c> and <c>ShapePositionEmbedder</c>
/// (<c>blocks.py</c>, <c>mm_latent.py</c>). Each coordinate gets <c>sin</c> then <c>cos</c> of <c>coordinate · f_i</c> with
/// <c>f_i = 1 / 10000^(i / freqDim)</c>, <c>freqDim = channels / inChannels / 2</c>; the per-coordinate blocks are laid side by side and the row is
/// zero-padded to <c>channels</c>. The released checkpoint stores the shape grid as a buffer; this computes it for when a conversion drops it.
/// </summary>
public static class SamPositionEmbedding
{
    private const float BasePeriod = 10000f;

    /// <summary>The embedding of <paramref name="coords"/> (row-major, <paramref name="inChannels"/> integers per position) as F32 <c>[n, channels]</c>.</summary>
    public static Tensor Absolute(int channels, int inChannels, ReadOnlySpan<int> coords)
    {
        if (channels <= 0 || inChannels <= 0)
            throw new ArgumentException("The channel counts must be positive.");
        if (coords.Length % inChannels != 0)
            throw new ArgumentException($"Coordinates are triples of {inChannels}; got {coords.Length} values.", nameof(coords));
        int freqDim = channels / inChannels / 2;
        int n = coords.Length / inChannels;
        Tensor result = new(new TensorShape(n, channels), DType.F32);
        Span<float> output = result.AsSpan<float>();
        float[] freqs = new float[freqDim];
        for (int i = 0; i < freqDim; i++)
            freqs[i] = 1f / MathF.Pow(BasePeriod, (float)i / freqDim);
        for (int row = 0; row < n; row++)
        {
            Span<float> target = output.Slice(row * channels, channels);
            for (int axis = 0; axis < inChannels; axis++)
            {
                float coordinate = coords[row * inChannels + axis];
                for (int i = 0; i < freqDim; i++)
                {
                    float angle = coordinate * freqs[i];
                    target[axis * 2 * freqDim + i] = MathF.Sin(angle);
                    target[axis * 2 * freqDim + freqDim + i] = MathF.Cos(angle);
                }
            }
        }

        return result;
    }

    /// <summary>The 3D grid embedding of <c>ShapePositionEmbedder</c>: the cells of a <c>resolution / patchSize</c> cube in <c>ij</c> order (x slowest),
    /// F32 <c>[(resolution / patchSize)^3, channels]</c>.</summary>
    public static Tensor ShapeGrid(int channels, int resolution, int patchSize)
    {
        if (resolution <= 0 || patchSize <= 0 || resolution % patchSize != 0)
            throw new ArgumentException("The resolution must be a positive multiple of the patch size.");
        int side = resolution / patchSize;
        int[] coords = new int[side * side * side * 3];
        int index = 0;
        for (int x = 0; x < side; x++)
        {
            for (int y = 0; y < side; y++)
            {
                for (int z = 0; z < side; z++)
                {
                    coords[index++] = x;
                    coords[index++] = y;
                    coords[index++] = z;
                }
            }
        }

        return Absolute(channels, 3, coords);
    }
}
