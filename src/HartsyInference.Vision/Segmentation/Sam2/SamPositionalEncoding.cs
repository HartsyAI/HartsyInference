using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>SAM's <c>PositionEmbeddingRandom</c>: maps a 2D coordinate to a <c>2*numPosFeats</c> vector via a
/// fixed Gaussian random projection followed by sin/cos. Coordinates are normalized to [0,1] (cell-centered),
/// mapped to [-1,1], projected through the Gaussian matrix, scaled by 2π, then encoded as
/// <c>[sin(·), cos(·)]</c>. Deterministic given the loaded Gaussian matrix.</summary>
public sealed unsafe class SamPositionalEncoding
{
    private readonly float[] _gaussian; // [2, numPosFeats] row-major
    private readonly int _numPosFeats;

    /// <summary>Output dimensionality (<c>2 * numPosFeats</c>).</summary>
    public int EmbedDim => _numPosFeats * 2;

    /// <summary>Creates the encoder from a <c>[2, numPosFeats]</c> Gaussian matrix (SAM's
    /// <c>positional_encoding_gaussian_matrix</c>).</summary>
    public SamPositionalEncoding(Tensor gaussianMatrix)
    {
        if (gaussianMatrix.Shape.Rank != 2 || gaussianMatrix.Shape[0] != 2)
            throw new ArgumentException($"Gaussian matrix must be [2, numPosFeats]; got {gaussianMatrix.Shape}.", nameof(gaussianMatrix));

        _numPosFeats = (int)gaussianMatrix.Shape[1];
        Tensor f32 = gaussianMatrix.DType == DType.F32 ? gaussianMatrix : gaussianMatrix.CastTo(DType.F32);
        _gaussian = new float[2 * _numPosFeats];
        f32.AsReadOnlySpan<float>().CopyTo(_gaussian);
        if (!ReferenceEquals(f32, gaussianMatrix)) f32.Dispose();
    }

    /// <summary>Encodes a pixel coordinate (normalized by the image size) into a <c>[EmbedDim]</c> vector written
    /// to <paramref name="destination"/>. <paramref name="cellCenter"/> adds the +0.5 the reference uses for the
    /// dense grid (<c>get_dense_pe</c>); sparse prompt points pass <c>false</c> (<c>forward_with_coords</c>).</summary>
    public void Encode(float x, float y, int imageWidth, int imageHeight, Span<float> destination, bool cellCenter = true)
    {
        if (destination.Length < EmbedDim)
            throw new ArgumentException($"Destination must hold at least {EmbedDim} floats.", nameof(destination));

        // Normalize to [0,1] (cell-centered for the dense grid only), then to [-1,1].
        float cx = cellCenter ? x + 0.5f : x;
        float cy = cellCenter ? y + 0.5f : y;
        float nx = cx / imageWidth * 2f - 1f;
        float ny = cy / imageHeight * 2f - 1f;

        for (int f = 0; f < _numPosFeats; f++)
        {
            // proj = [nx, ny] · gaussian[:, f]  ×  2π
            float proj = (nx * _gaussian[f] + ny * _gaussian[_numPosFeats + f]) * 2f * MathF.PI;
            destination[f] = MathF.Sin(proj);
            destination[_numPosFeats + f] = MathF.Cos(proj);
        }
    }

    /// <summary>Encodes a single coordinate into a fresh <c>float[EmbedDim]</c>.</summary>
    public float[] Encode(float x, float y, int imageWidth, int imageHeight)
    {
        float[] outv = new float[EmbedDim];
        Encode(x, y, imageWidth, imageHeight, outv);
        return outv;
    }
}
