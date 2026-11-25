namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>The three state updates of the Duo flow sampler, on flat F32 spans. Flow time runs noise (0) to data (1), so <c>x(t) = t * clean + (1 - t) * noise</c> and the velocity is <c>clean - noise</c>
/// (upstream <c>duo/models/realization/schedulers/flow.py</c> @068e357).</summary>
public static class DuoSampler
{
    /// <summary>Euler step: <c>dst = noisy + (nextT - t) * velocity</c>. The causal student integrates with this.</summary>
    public static void Euler(ReadOnlySpan<float> noisy, ReadOnlySpan<float> velocity, float t, float nextT, Span<float> dst)
    {
        CheckLengths(noisy, velocity, dst);
        float step = nextT - t;
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] = noisy[i] + step * velocity[i];
        }
    }

    /// <summary>The clean estimate: <c>dst = noisy + (1 - t) * prediction</c>. The distilled generator takes this step, then re-noises.</summary>
    public static void ToX0(ReadOnlySpan<float> noisy, ReadOnlySpan<float> prediction, float t, Span<float> dst)
    {
        CheckLengths(noisy, prediction, dst);
        float remaining = 1f - t;
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] = noisy[i] + remaining * prediction[i];
        }
    }

    /// <summary>Re-noise to time <paramref name="t"/>: <c>dst = clean * t + noise * (1 - t)</c>.</summary>
    public static void AddNoise(ReadOnlySpan<float> clean, ReadOnlySpan<float> noise, float t, Span<float> dst)
    {
        CheckLengths(clean, noise, dst);
        float remaining = 1f - t;
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] = clean[i] * t + noise[i] * remaining;
        }
    }

    private static void CheckLengths(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
    {
        if (a.Length != b.Length || a.Length != dst.Length)
        {
            throw new ArgumentException($"Sampler spans differ: {a.Length}, {b.Length}, {dst.Length}.");
        }
    }
}
