namespace HartsyInference.Diffusion.Schedulers;

/// <summary>SANA-WM <c>flow_euler_ltx</c> sigma schedule, reproducing diffusers <c>FlowMatchEulerDiscreteScheduler</c>
/// <c>.set_timesteps</c> (static shift, no terminal stretch) as driven by <c>retrieve_timesteps</c>. diffusers builds its
/// linspace from the sigma range it already shifted at construction, then shifts again, so the shift is applied twice.
/// Operation order and precision follow diffusers exactly: endpoints in float32, linspace and shift in float64, cast to
/// float32 at the end.</summary>
public static class SanaWmSchedule
{
    private const double NumTrainTimesteps = 1000.0;

    /// <summary>Builds <c>steps + 1</c> descending sigmas ending at 0, and the <c>steps</c> timesteps (<c>sigma * 1000</c>).
    /// Pass the first <c>steps</c> sigmas to <see cref="FlowMatchEulerDiscreteScheduler.SetSigmas"/>, which appends the
    /// terminal 0 itself.</summary>
    /// <param name="steps">Denoise step count, at least 1.</param>
    /// <param name="shift">Static schedule shift. The upstream default is 9.8 and the demo uses 8.0.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="steps"/> is below 1, or <paramref name="shift"/> is
    /// not a positive finite number.</exception>
    public static (float[] Sigmas, float[] Timesteps) Build(int steps, double shift)
    {
        if (steps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "A SANA-WM schedule needs at least one step.");
        }
        if (!double.IsFinite(shift) || shift <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(shift), shift, "Sigma shift must be a positive finite number.");
        }

        float shift32 = (float)shift;
        float shiftMinusOne32 = (float)(shift - 1.0);
        float sigmaMax = ShiftFloat32(1.0f, shift32, shiftMinusOne32);
        float sigmaMin = ShiftFloat32(1.0f / (float)NumTrainTimesteps, shift32, shiftMinusOne32);
        double start = sigmaMax * NumTrainTimesteps;
        double stop = sigmaMin * NumTrainTimesteps;
        double stride = steps > 1 ? (stop - start) / (steps - 1) : 0.0;

        float[] sigmas = new float[steps + 1];
        float[] timesteps = new float[steps];
        for (int i = 0; i < steps; i++)
        {
            double grid = steps > 1 && i == steps - 1 ? stop : i * stride + start;
            double t = grid / NumTrainTimesteps;
            sigmas[i] = (float)(shift * t / (1.0 + (shift - 1.0) * t));
            timesteps[i] = sigmas[i] * (float)NumTrainTimesteps;
        }
        sigmas[steps] = 0.0f;
        return (sigmas, timesteps);
    }

    private static float ShiftFloat32(float sigma, float shift, float shiftMinusOne)
    {
        float scaled = shift * sigma;
        return scaled / (1.0f + shiftMinusOne * sigma);
    }
}
