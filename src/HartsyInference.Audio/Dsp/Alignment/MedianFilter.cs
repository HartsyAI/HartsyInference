namespace HartsyInference.Audio.Dsp.Alignment;

/// <summary>Sliding median along the columns of a row-major matrix.</summary>
public static class MedianFilter
{
    /// <summary>Replaces every element of each row of the <c>[rows, cols]</c> matrix with the median of the
    /// <paramref name="width"/> elements centred on it, mirroring the row at both edges without repeating the edge
    /// sample. A row no longer than half the window is left as it is.</summary>
    /// <param name="width">The odd window length; 1 leaves the data untouched.</param>
    public static void Rows(Span<float> data, int rows, int cols, int width)
    {
        if (width < 1 || (width & 1) == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), width, "The window must be odd and at least 1.");
        }
        if (data.Length != rows * cols)
        {
            throw new ArgumentException($"Data has {data.Length} entries but {rows} x {cols} needs {rows * cols}.", nameof(data));
        }
        int half = width / 2;
        if (width == 1 || cols <= half)
        {
            return;
        }

        float[] source = new float[cols];
        Span<float> window = stackalloc float[width];
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = data.Slice(r * cols, cols);
            row.CopyTo(source);
            for (int c = 0; c < cols; c++)
            {
                for (int k = 0; k < width; k++)
                {
                    window[k] = source[Reflect(c + k - half, cols)];
                }
                window.Sort();
                row[c] = window[half];
            }
        }
    }

    /// <summary>Mirrors an index that ran past either end of a row of <paramref name="length"/> samples back inside it.</summary>
    private static int Reflect(int index, int length)
    {
        if (index < 0)
        {
            return -index;
        }
        return index >= length ? 2 * (length - 1) - index : index;
    }
}
