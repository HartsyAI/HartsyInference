namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>F5-TTS Sway-Sampling scheduler: standard flow-matching Euler over a uniform timestep grid in <c>[0, 1]</c>, with a per-step "sway" warp that biases more samples toward the noise-end of the trajectory. Matches <c>F5TTS.model.cfm.CFM.sample</c> in the upstream repo.</summary>
// Sway formula:
//   u = linspace(0, 1, steps + 1)               # uniform grid
//   if sway_coef != 0:
//     u = u + sway_coef * (cos(pi/2 * u) - 1 + u)
//   sigmas = u                                   # flow-matching: sigma == t
// F5-TTS v1 default sway_coef = -1.0 shifts mass toward t=0 (the noise end), which empirically helps
// quality at low NFE counts.
// This is a CPU-side scheduler — produces a float[] sigma table and a matching float[] of step deltas
// that the pipeline consumes in its denoise loop alongside the DiT vector-field predictions.
public sealed class F5SwaySamplingScheduler
{
    /// <summary>The N+1 timestep values from 0 (noise) through 1 (data), post-sway; use successive deltas (<see cref="Deltas"/>) for Euler step sizes.</summary>
    public float[] Timesteps { get; }

    /// <summary>The N step deltas <c>t[i+1] - t[i]</c>.</summary>
    public float[] Deltas { get; }

    /// <summary>Number of function evaluations (NFE) = <c>Timesteps.Length - 1</c>.</summary>
    public int Steps => Timesteps.Length - 1;

    /// <summary><paramref name="steps"/> is the NFE (default 32 for F5-TTS v1); <paramref name="swayCoef"/> defaults to -1.0 (F5-TTS recommended).</summary>
    public F5SwaySamplingScheduler(int steps = 32, float swayCoef = -1.0f)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));

        float[] u = new float[steps + 1];
        for (int i = 0; i <= steps; i++) u[i] = (float)i / steps;

        if (swayCoef != 0f)
        {
            for (int i = 0; i <= steps; i++)
            {
                float uu = u[i];
                float sway = swayCoef * (MathF.Cos(MathF.PI * 0.5f * uu) - 1f + uu);
                u[i] = uu + sway;
            }
        }

        Timesteps = u;
        Deltas = new float[steps];
        for (int i = 0; i < steps; i++) Deltas[i] = u[i + 1] - u[i];
    }
}
