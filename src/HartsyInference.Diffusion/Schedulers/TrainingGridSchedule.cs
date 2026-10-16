namespace HartsyInference.Diffusion.Schedulers;

/// <summary>A shifted flow-match schedule snapped to the discrete grid a model was trained on, as MOSS-Image's
/// <c>MOSSImageScheduler</c> builds it: <c>count</c> points spaced linearly from 1 down to the shifted grid's lowest
/// training point, shifted by <c>σ = s·v/(1 + (s − 1)·v)</c>, then each moved to the nearest training sigma (ties to the
/// lower one) and consecutive duplicates dropped. Each grid point is computed in F64 and rounded to F32 once. The
/// reference builds them in F32 with <c>torch.linspace</c>, whose last bit depends on the CPU (FMA contraction and
/// SIMD width change which partial products round), so no fixed F32 recipe matches every machine; one rounding of the
/// exact value lands the training grid on <c>j/1000</c> and reproduces the reference's published goldens
/// (SGLang <c>test_v2_schedule_and_shift_override</c>). Feed the result to
/// <see cref="FlowMatchEulerDiscreteScheduler.SetSigmas"/>, which appends the terminal 0.</summary>
public static class TrainingGridSchedule
{
    /// <summary>The snapped descending sigmas for <paramref name="steps"/> denoise steps, excluding the terminal 0. Fewer
    /// than <paramref name="steps"/> come back when two steps snap to the same training point.</summary>
    /// <param name="steps">Requested denoise steps.</param>
    /// <param name="shift">Inference time shift.</param>
    /// <param name="trainSteps">Training grid size (the scheduler's <c>num_train_timesteps</c>).</param>
    /// <param name="trainingShift">Shift the training grid was built with.</param>
    public static float[] Sigmas(int steps, float shift, int trainSteps = 1000, float trainingShift = 1f)
    {
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "A schedule needs at least one step.");
        if (trainSteps < 2)
            throw new ArgumentOutOfRangeException(nameof(trainSteps), trainSteps, "A training grid needs at least two points.");
        if (!float.IsFinite(shift) || shift <= 0f)
            throw new ArgumentOutOfRangeException(nameof(shift), shift, "The time shift must be positive and finite.");
        if (!float.IsFinite(trainingShift) || trainingShift <= 0f)
            throw new ArgumentOutOfRangeException(nameof(trainingShift), trainingShift, "The training shift must be positive and finite.");

        float[] wanted = ShiftedGrid(steps, shift, trainSteps);
        float[] training = ShiftedGrid(trainSteps, trainingShift, trainSteps);
        Array.Reverse(training);
        List<float> snapped = new List<float>(steps);
        foreach (float sigma in wanted)
        {
            float value = training[Nearest(training, sigma)];
            if (snapped.Count == 0 || snapped[^1] != value)
                snapped.Add(value);
        }
        return [.. snapped];
    }

    /// <summary><c>count</c> linear points from 1 to <c>shift/(total + shift − 1)</c>, shifted, each rounded once: the
    /// point the lowest training sigma maps to under this shift, so the grid ends on the training grid's floor.</summary>
    private static float[] ShiftedGrid(int count, float shift, int total)
    {
        double lower = shift / (total + (double)shift - 1.0);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            double v = count == 1 ? 1.0 : 1.0 + (lower - 1.0) * i / (count - 1);
            values[i] = (float)(shift * v / (1.0 + (shift - 1.0) * v));
        }
        return values;
    }

    /// <summary>Index of the ascending <paramref name="grid"/> point nearest <paramref name="sigma"/>, ties to the lower
    /// index (<c>torch.searchsorted</c> left side, then the closer neighbour).</summary>
    private static int Nearest(float[] grid, float sigma)
    {
        int upper = LowerBound(grid, sigma);
        upper = Math.Min(upper, grid.Length - 1);
        int lower = Math.Max(upper - 1, 0);
        return MathF.Abs(sigma - grid[lower]) <= MathF.Abs(grid[upper] - sigma) ? lower : upper;
    }

    private static int LowerBound(float[] grid, float value)
    {
        int lo = 0, hi = grid.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (grid[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}
