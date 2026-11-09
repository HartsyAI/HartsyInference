namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Row permutations that move a square token grid between row-major and window-major order.</summary>
public static class Sam3WindowOrder
{
    /// <summary>Returns, for each window-major row, the row-major token index it reads.</summary>
    /// <param name="grid">Token grid side.</param>
    /// <param name="window">Window side; must divide <paramref name="grid"/>.</param>
    public static int[] CreateGather(int grid, int window)
    {
        if (grid <= 0 || window <= 0 || grid % window != 0)
            throw new ArgumentException($"Grid {grid} must be a positive multiple of window {window}.");

        int perSide = grid / window;
        int[] gather = new int[grid * grid];
        int row = 0;
        for (int wy = 0; wy < perSide; wy++)
        {
            for (int wx = 0; wx < perSide; wx++)
            {
                for (int r = 0; r < window; r++)
                {
                    for (int c = 0; c < window; c++)
                    {
                        gather[row++] = (wy * window + r) * grid + wx * window + c;
                    }
                }
            }
        }

        return gather;
    }

    /// <summary>Returns the inverse permutation: for each row-major token, the window-major row holding it.</summary>
    /// <param name="permutation">A permutation of <c>0..n-1</c>.</param>
    public static int[] Invert(int[] permutation)
    {
        ArgumentNullException.ThrowIfNull(permutation);

        int[] inverse = new int[permutation.Length];
        for (int i = 0; i < permutation.Length; i++)
        {
            inverse[permutation[i]] = i;
        }

        return inverse;
    }
}
