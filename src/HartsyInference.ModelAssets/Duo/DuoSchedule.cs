namespace HartsyInference.ModelAssets.Duo;

/// <summary>The fixed flow-time grids of the Duo Realization stages, which is all the AdaLN modulation tables need: the DiT's modulation depends only on a scalar time and the row's modality, so a stage with a fixed schedule has a handful of distinct times.
/// Times run noise (0) to data (1), unscaled. The shift is applied to the noise level <c>sigma = 1 - t</c> as <c>sigma' = shift * sigma / (1 + (shift - 1) * sigma)</c>; this form reproduces the distinct distill times the research notes derived
/// (<c>0, 0.0625, 0.1667, 0.375</c> for shift 5) and is not yet checked against upstream's <c>grid()</c>.</summary>
public static class DuoSchedule
{
    /// <summary>The 4-step DMD generator: <c>grid(5, shift 5.0)</c>.</summary>
    public const int DistillSteps = 4;

    public const double DistillShift = 5.0;

    /// <summary>The 50-step causal student: <c>grid(51, shift 1.0)</c>.</summary>
    public const int CausalSteps = 50;

    public const double CausalShift = 1.0;

    /// <summary>The time of a clean (committed) chunk, which equals the last grid point.</summary>
    public const float CleanTime = 1.0f;

    /// <summary>The <paramref name="steps"/> + 1 grid points, starting at 0 and ending at 1.</summary>
    public static double[] Grid(int steps, double shift)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        if (!(shift > 0) || !double.IsFinite(shift)) throw new ArgumentOutOfRangeException(nameof(shift), shift, "The shift must be positive and finite.");
        double[] times = new double[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            double sigma = 1.0 - (double)i / steps;
            double shifted = shift * sigma / (1.0 + (shift - 1.0) * sigma);
            times[i] = 1.0 - shifted;
        }
        times[steps] = 1.0;
        return times;
    }

    /// <summary>The distinct times a stage conditions on: every grid point, the last being the clean time. Five for the distill, 51 for the causal student.</summary>
    public static float[] AdalnTimes(int steps, double shift)
    {
        double[] grid = Grid(steps, shift);
        float[] times = new float[grid.Length];
        for (int i = 0; i < times.Length; i++) times[i] = (float)grid[i];
        return times;
    }
}
