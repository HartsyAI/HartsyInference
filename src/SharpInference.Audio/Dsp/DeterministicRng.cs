namespace SharpInference.Audio.Dsp;

/// <summary>A small deterministic, allocation-free PRNG (xorshift32 + Box-Muller) shared by every model
/// that needs reproducible noise — NSF harmonic sources, flow-matching / diffusion samplers, the
/// speech-token sampler. Centralized so a fixed seed gives identical output across models and there is
/// one RNG to reason about, not a copy per model. Thread the <c>uint</c> state by ref through a loop.</summary>
public static class DeterministicRng
{
    /// <summary>Derives a non-zero state from an integer seed (the seeding convention every model used).</summary>
    public static uint Seed(int seed) => unchecked((uint)seed * 2654435761u + 0x9E3779B9u) | 1u;

    /// <summary>Advances the state and returns a uniform sample in <c>[0, 1)</c> with 24-bit resolution.</summary>
    public static float NextUniform(ref uint state)
    {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return (state & 0xFFFFFF) / 16777216f;
    }

    /// <summary>Advances the state twice and returns one standard-normal sample via Box-Muller.</summary>
    public static float NextGaussian(ref uint state)
    {
        float u1 = NextUniform(ref state);
        float u2 = NextUniform(ref state);
        if (u1 < 1e-7f) u1 = 1e-7f;
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }
}
