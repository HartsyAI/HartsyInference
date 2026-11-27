namespace HartsyInference.ThreeD.Geometry;

/// <summary>The size a dense marching-cubes field can have: its cube of samples is held in one managed array, which holds at most <see cref="Array.MaxLength"/> elements.</summary>
public static class GridLimit
{
    /// <summary>The most samples per axis whose cube fits one array: the cube root of <see cref="Array.MaxLength"/>, rounded down (a test pins it against the array limit).</summary>
    public const int LargestAxis = 1290;

    /// <summary>Whether a field of <paramref name="axisPoints"/> samples per axis fits one array. A negative axis does not fit; the axis is capped before cubing, so a huge value cannot wrap the product.</summary>
    public static bool Fits(long axisPoints)
    {
        if (axisPoints < 0)
            return false;
        long axis = Math.Min(axisPoints, LargestAxis + 1L);
        return axis * axis * axis <= Array.MaxLength;
    }

    /// <summary>Checks a dense grid whose field has <paramref name="grid"/> samples per axis; 0 or less takes the model's default and passes.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The cube does not fit one array; the parameter is <c>grid</c>.</exception>
    public static void RequireDense(int grid)
    {
        if (grid <= 0 || Fits(grid))
            return;
        throw new ArgumentOutOfRangeException(
            "grid", grid, $"A grid of {grid} needs a field too large for one array; the largest that fits one is {LargestAxis} (host memory runs out sooner).");
    }
}
