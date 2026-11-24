namespace HartsyInference.Diffusion.Schedulers;

/// <summary>The flow-matching noise grid of FastFrames and the few-step Causal Forcing generators it builds on: a rectified-flow
/// schedule <c>x_t = (1 - σ)·x_0 + σ·noise</c> whose σ values are a 1000-entry grid, <c>linspace(1, 0, 1001)</c> without its last
/// point, shifted by <c>shift·σ / (1 + (shift - 1)·σ)</c>, with training timesteps <c>σ·1000</c>. A requested timestep (the
/// generators ask for values such as 833.33 or 937.5) is not interpolated: it maps to the grid entry whose timestep is nearest, and
/// that entry's σ is used both to noise a frame and to turn a flow prediction into a clean-latent estimate
/// (<c>x_0 = x_t - σ·flow</c>). It differs from <see cref="SelfForcingFlowSchedule"/>, which takes its timesteps verbatim as
/// sigmas and applies no shift. Follows <c>fastframes/wan/scheduler.py</c> (ml-fastframes a65c592).</summary>
public sealed class FastFramesFlowSchedule
{
    /// <summary>Entries of the σ grid, and the timestep scale.</summary>
    public const int GridSize = 1000;

    private readonly float[] _sigmas;
    private readonly float[] _timesteps;

    /// <summary>Builds the grid for <paramref name="shift"/> (5 for the released models).</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="shift"/> is not positive.</exception>
    public FastFramesFlowSchedule(float shift = 5f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(shift, 0f);
        Shift = shift;
        _sigmas = new float[GridSize];
        _timesteps = new float[GridSize];
        for (int i = 0; i < GridSize; i++)
        {
            // torch.linspace(1, 0, 1001)[:-1]: evaluated in double, stored as float32, as the reference stores it.
            double s = 1.0 - (double)i / GridSize;
            _sigmas[i] = (float)(shift * s / (1.0 + (shift - 1.0) * s));
            _timesteps[i] = _sigmas[i] * GridSize;
        }
    }

    /// <summary>The shift this grid was built with.</summary>
    public float Shift { get; }

    /// <summary>The grid entry whose timestep is nearest to <paramref name="timestep"/>; the first such entry on a tie, as
    /// <c>argmin</c> picks it.</summary>
    public int GridIndex(float timestep)
    {
        int best = 0;
        float bestDistance = MathF.Abs(_timesteps[0] - timestep);
        for (int i = 1; i < GridSize; i++)
        {
            float distance = MathF.Abs(_timesteps[i] - timestep);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = i;
            }
        }
        return best;
    }

    /// <summary>The σ of the grid entry nearest to <paramref name="timestep"/>; 0 returns the grid's last (smallest) σ, not 0, as
    /// the reference's lookup does.</summary>
    public float Sigma(float timestep) => _sigmas[GridIndex(timestep)];

    /// <summary>The training timestep of grid entry <paramref name="index"/>.</summary>
    public float Timestep(int index) => _timesteps[index];
}
