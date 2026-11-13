namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MiniMax-H3's dual flow-match schedule. The sampler integrates the VIDEO sigma; the audio stream lives on
/// its own shifted schedule derived from it in closed form, and its velocity is scaled by the map's derivative so a
/// single stock integrator over the flat audio+video pack still solves each stream's true ODE.</summary>
public static class MiniMaxH3Schedule
{
    /// <summary>Video flow shift the sampler runs on.</summary>
    public const float DefaultShiftVideo = 12.0f;

    /// <summary>Audio flow shift; the audio stream is denoised on this schedule.</summary>
    public const float DefaultShiftAudio = 3.0f;

    /// <summary>Timestep the visual conditioning rows are pinned near — they are already clean.</summary>
    public const float VisualCondTimestep = 0.999f;

    /// <summary>Timestep the audio conditioning rows are pinned at.</summary>
    public const float AudioCondTimestep = 1.0f;

    /// <summary>Re-expresses <paramref name="sigma"/> from one shift onto another by inverting
    /// <c>sigma = s*b / (1 + (s-1)*b)</c> back to the unshifted grid and re-applying the target shift.</summary>
    public static double ShiftSigma(double sigma, double fromShift, double toShift)
    {
        double b = Base(sigma, fromShift);
        return toShift * b / (1.0 + (toShift - 1.0) * b);
    }

    /// <summary><c>d(sigma_to)/d(sigma_from)</c> at the same unshifted point. Scaling a stream's velocity by this makes
    /// the flat ODE the sampler integrates on the from-schedule equal that stream's ODE on its own schedule.</summary>
    public static double ShiftSlope(double sigma, double fromShift, double toShift)
    {
        double b = Base(sigma, fromShift);
        double num = toShift * Math.Pow(1.0 + (fromShift - 1.0) * b, 2.0);
        double den = fromShift * Math.Pow(1.0 + (toShift - 1.0) * b, 2.0);
        return num / den;
    }

    /// <summary>Applies a flow shift to an unshifted sigma in [0,1].</summary>
    public static double ApplyShift(double baseSigma, double shift) =>
        shift * baseSigma / (1.0 + (shift - 1.0) * baseSigma);

    /// <summary>The descending video sigma grid: <c>linspace(1, 1/steps, steps)</c> with the video shift applied,
    /// then a trailing 0 so the last step integrates to a clean latent.</summary>
    public static double[] VideoSigmas(int steps, double shiftVideo)
    {
        if (steps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(steps), "MiniMax-H3 needs at least one step.");
        }
        double[] sigmas = new double[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            double raw = 1.0 - i * (1.0 - 1.0 / steps) / Math.Max(1, steps - 1);
            sigmas[i] = ApplyShift(raw, shiftVideo);
        }
        sigmas[steps] = 0.0;
        return sigmas;
    }

    /// <summary>The MiniMax-H3-Image sigma schedule that ComfyUI's stock <c>simple</c> scheduler produces at
    /// <paramref name="steps"/> steps. ComfyUI reads the checkpoint grid piecewise-linearly in the timestep
    /// (<c>comfy/ldm/minimax/sampling.py</c>), samples its 1000-entry table at <c>int(x * 1000 / steps)</c> from the top
    /// (<c>comfy/samplers.py</c>, <c>simple_scheduler</c>), and appends 0. At six steps the timesteps are 1000, 834, 667,
    /// 500, 334 and 167, so four of the sigmas sit between grid points. The heads are still chosen by nearest grid point.
    /// The schedule values follow ComfyUI's simple scheduler; the arithmetic is in double, so they can differ from ComfyUI's
    /// float32 by about one ulp.</summary>
    public static double[] ImageSigmas(IReadOnlyList<double> grid, int steps = 6)
    {
        ArgumentNullException.ThrowIfNull(grid);
        if (grid.Count < 2 || grid[0] != 1.0 || grid[^1] != 0.0)
        {
            throw new ArgumentException("The MiniMax-H3-Image grid must run from 1.0 to 0.0 with at least one step.", nameof(grid));
        }
        if (steps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "MiniMax-H3-Image needs at least one step.");
        }

        double[] sigmas = new double[steps + 1];
        for (int x = 0; x < steps; x++)
        {
            int timestep = 1000 - (int)(x * (1000.0 / steps));
            sigmas[x] = GridSigmaAt(grid, timestep);
        }
        sigmas[steps] = 0.0;
        return sigmas;
    }

    /// <summary>ComfyUI's <c>MiniMaxH3ImageModelSampling.sigma</c>: the ascending grid read linearly between its points at
    /// <c>timestep / 1000</c> scaled to the segment count, with the last segment's end clamped.</summary>
    private static double GridSigmaAt(IReadOnlyList<double> grid, int timestep)
    {
        int segments = grid.Count - 1;
        double pos = Math.Clamp(timestep / 1000.0, 0.0, 1.0) * segments;
        int i = Math.Min((int)Math.Floor(pos), segments - 1);
        // The grid is stored descending; ComfyUI flips it, so ascending point k is grid[segments - k].
        double start = grid[segments - i];
        double end = grid[segments - (i + 1)];
        double weight = pos - i;
        // torch.lerp's two branches, so the rounding matches the reference.
        return weight < 0.5 ? start + weight * (end - start) : end - (end - start) * (1.0 - weight);
    }

    /// <summary>Per-token timesteps for one step: video/text follow <c>1 - sigma_v</c>, audio follows its own shifted
    /// schedule, and conditioning rows pin near 1 because they carry clean latents.</summary>
    public static (float Video, float Audio) Timesteps(double sigmaVideo, double shiftVideo, double shiftAudio)
    {
        double clamped = Math.Max(sigmaVideo, 1e-6);
        float tVideo = (float)(1.0 - clamped);
        float tAudio = (float)(1.0 - ShiftSigma(clamped, shiftVideo, shiftAudio));
        return (tVideo, tAudio);
    }

    private static double Base(double sigma, double shift) => sigma / (shift + sigma * (1.0 - shift));
}
