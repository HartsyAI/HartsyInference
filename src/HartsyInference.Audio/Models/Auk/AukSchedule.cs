using HartsyInference.Audio.Models.F5Tts;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AuK flow-matching sampling schedule: <c>x_t = (1-t)·x0 + t·x1</c>, Euler <c>x += v·dt</c> from t=0 (noise) to t=1, with optional classifier-free guidance.</summary>
public sealed class AukSchedule
{
    /// <summary>Guidance scales below this skip the unconditional forward entirely.</summary>
    public const float CfgEpsilon = 1e-5f;

    /// <summary>AuK-Flash's fixed 4-step timestep grid.</summary>
    public static ReadOnlySpan<float> FlashGrid => [0f, 0.07612049579620361f, 0.2928932309150696f, 0.6173166036605835f, 1f];

    /// <summary>The N+1 timesteps from 0 through 1.</summary>
    public float[] Timesteps { get; }

    /// <summary>The N step sizes <c>t[i+1] - t[i]</c>.</summary>
    public float[] Deltas { get; }

    /// <summary>Guidance scale; 0 means a single conditional forward per step.</summary>
    public float Cfg { get; }

    /// <summary>Number of function evaluations (Euler steps).</summary>
    public int Steps => Timesteps.Length - 1;

    /// <summary>True when each step needs a second, unconditional forward.</summary>
    public bool UsesCfg => Cfg >= CfgEpsilon;

    private AukSchedule(float[] timesteps, float cfg)
    {
        Timesteps = timesteps;
        Cfg = cfg;
        Deltas = new float[timesteps.Length - 1];
        for (int i = 0; i < Deltas.Length; i++) Deltas[i] = timesteps[i + 1] - timesteps[i];
    }

    /// <summary>AuK base: uniform grid warped by sway sampling (coefficient -1), CFG 2.0 and 32 steps by default.</summary>
    public static AukSchedule Base(int steps = 32, float cfg = 2f, float swayCoef = -1f)
    {
        if (cfg < 0f || !float.IsFinite(cfg)) throw new ArgumentOutOfRangeException(nameof(cfg));
        return new AukSchedule(new F5SwaySamplingScheduler(steps, swayCoef).Timesteps, cfg);
    }

    /// <summary>AuK-Flash: fixed 4-step grid, no guidance, no sway.</summary>
    public static AukSchedule Flash() => new(FlashGrid.ToArray(), 0f);

    /// <summary>One Euler step in place: <c>x += v·dt</c>.</summary>
    public static void EulerStep(Span<float> x, ReadOnlySpan<float> v, float dt)
    {
        if (x.Length != v.Length) throw new ArgumentException("x and v must have the same length.", nameof(v));
        for (int i = 0; i < x.Length; i++) x[i] += v[i] * dt;
    }

    /// <summary>CFG combine in place on the conditional velocity: <c>v = v_c + (v_c - v_u)·cfg</c>; a no-op when <paramref name="cfg"/> is below <see cref="CfgEpsilon"/>.</summary>
    public static void CfgCombine(Span<float> vCond, ReadOnlySpan<float> vUncond, float cfg)
    {
        if (vCond.Length != vUncond.Length) throw new ArgumentException("Velocities must have the same length.", nameof(vUncond));
        if (cfg < CfgEpsilon) return;
        for (int i = 0; i < vCond.Length; i++) vCond[i] += (vCond[i] - vUncond[i]) * cfg;
    }
}
