namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Block-scaled e4m3 dequantization of Hy4 Flash weights: element <c>[..., r, c]</c> of a matrix (or each expert of a stack) is multiplied by the scale of its
/// <paramref name="block"/> by <paramref name="block"/> block; the last block row and column may be partial.</summary>
public static class Hy4FlashFp8
{
    /// <summary>The released block edge.</summary>
    public const int Block = 128;

    /// <summary>Decodes <paramref name="codes"/> of shape <paramref name="shape"/> (the last two dimensions are the matrix, any leading ones are experts) into F32.</summary>
    /// <param name="scale">The block grid, <c>[lead, ceil(out / block), ceil(in / block)]</c> row-major, as F32.</param>
    /// <exception cref="ArgumentException">The shape has fewer than two dimensions or does not match the code count.</exception>
    /// <exception cref="InvalidDataException">The scale grid does not have the size the shape implies.</exception>
    public static float[] DequantizeBlocks(ReadOnlySpan<byte> codes, ReadOnlySpan<long> shape, ReadOnlySpan<float> scale, int block = Block)
    {
        float[] values = new float[codes.Length];
        DequantizeBlocks(codes, shape, scale, values, block);
        return values;
    }

    /// <summary>Decodes into <paramref name="destination"/>, one F32 value per code, so a caller can write straight into its tensor.</summary>
    /// <exception cref="ArgumentException">The shape, or the destination, does not match the code count.</exception>
    /// <exception cref="InvalidDataException">The scale grid does not have the size the shape implies.</exception>
    public static void DequantizeBlocks(ReadOnlySpan<byte> codes, ReadOnlySpan<long> shape, ReadOnlySpan<float> scale, Span<float> destination, int block = Block)
    {
        if (shape.Length < 2) throw new ArgumentException("An FP8 block-scaled weight has at least two dimensions.", nameof(shape));
        ArgumentOutOfRangeException.ThrowIfLessThan(block, 1);
        long rows = shape[^2], cols = shape[^1], lead = 1;
        for (int i = 0; i < shape.Length - 2; i++) lead *= shape[i];
        if (codes.Length != lead * rows * cols) throw new ArgumentException($"{codes.Length} codes do not fill shape with {lead * rows * cols} elements.", nameof(codes));
        if (destination.Length != codes.Length) throw new ArgumentException($"The destination holds {destination.Length} values for {codes.Length} codes.", nameof(destination));
        long gridRows = (rows + block - 1) / block, gridCols = (cols + block - 1) / block;
        if (scale.Length != lead * gridRows * gridCols)
            throw new InvalidDataException($"The scale grid holds {scale.Length} values; a [{lead}, {gridRows}, {gridCols}] grid is required.");
        for (long e = 0; e < lead; e++)
            for (long r = 0; r < rows; r++)
                for (long c = 0; c < cols; c++)
                {
                    int index = checked((int)((e * rows + r) * cols + c));
                    float s = scale[checked((int)((e * gridRows + r / block) * gridCols + c / block))];
                    destination[index] = Hy4FlashFp8Rounding.Decode(codes[index]) * s;
                }
    }
}
