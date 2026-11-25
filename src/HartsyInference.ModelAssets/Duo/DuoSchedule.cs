namespace HartsyInference.ModelAssets.Duo;

/// <summary>The fixed flow-time grids of the Duo Realization stages, which is all the AdaLN modulation tables need: the DiT's modulation depends only on a scalar time and the row's modality, so a stage with a fixed schedule has a handful of distinct times.
/// Times run noise (0) to data (1), unscaled. This is upstream's <c>grid</c> (<c>duo/models/realization/schedulers/flow.py</c> @068e357): <c>t = linspace(0, 1, points)</c>, <c>1 - shift * (1 - t) / (1 + (shift - 1) * (1 - t))</c>, in torch's default float32.
/// The sampler passes <c>grid[index]</c> as the time of every noisy block at step <c>index</c> and 1.0 for committed (clean) chunks; the last grid point, 1.0, equals the clean time.</summary>
public static class DuoSchedule
{
    /// <summary>The 4-step DMD generator: <c>grid(5, shift 5.0)</c>.</summary>
    public const int DistillSteps = 4;

    public const double DistillShift = 5.0;

    /// <summary>The 50-step causal student: <c>grid(51, shift 1.0)</c>.</summary>
    public const int CausalSteps = 50;

    public const double CausalShift = 1.0;

    /// <summary>The time of a clean (committed) chunk in inference.</summary>
    public const float CleanTime = 1.0f;

    /// <summary>The <paramref name="steps"/> + 1 grid points, evaluated in F32 the way torch evaluates them. Five times for the distill, 51 for the causal student; a consumer addresses a table row by step index, not by comparing floats.</summary>
    public static float[] Grid(int steps, double shift)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        if (!(shift > 0) || !double.IsFinite(shift)) throw new ArgumentOutOfRangeException(nameof(shift), shift, "The shift must be positive and finite.");
        int points = steps + 1;
        float step = 1f / (points - 1);
        float shiftF = (float)shift, shiftMinusOne = (float)(shift - 1.0);
        float[] times = new float[points];
        for (int i = 0; i < points; i++)
        {
            // torch.linspace counts from the start for the first half of the points and back from the end for the rest.
            float t = i < points / 2 ? step * i : 1f - step * (points - i - 1);
            float noise = 1f - t;
            times[i] = 1f - shiftF * noise / (1f + shiftMinusOne * noise);
        }
        return times;
    }
}
