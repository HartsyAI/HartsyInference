using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>Integrates YuE2's acoustic flow: a fixed-step midpoint solver running <c>t</c> from 1 down to <c>dt</c>.
/// The release protocol is 32 steps and midpoint; both are checkpoint facts rather than quality dials, so the only
/// knob exposed is the step count.</summary>
/// <remarks>The timestep handed to the model is <c>logit(t)</c>, not <c>t</c>, computed in double precision and
/// clamped to ±20 (so <c>t = 1</c> is representable). <see cref="Yue2AcousticTransformer"/> takes the sigmoid back
/// on its side, because the checkpoint's <c>timestep_shift</c> warp is defined on the model's own input.</remarks>
public static class Yue2FlowSolver
{
    /// <summary>Solves one acoustic chunk, returning <c>[frames, 64]</c> latents.</summary>
    /// <param name="noise">The chunk's initial state, <c>[frames, 64]</c>. The caller draws this — for a whole song
    /// the reference draws one frame-major F32 tensor on the host and slices it per chunk, so drawing per chunk
    /// changes every result.</param>
    /// <param name="arPrefix">Per-layer post-RoPE K/V from this chunk's AR prefill. Invariant across every step,
    /// which is why the prefill happens once outside the loop.</param>
    public static float[] Solve(IBackend backend, Yue2AcousticTransformer transformer, ReadOnlySpan<float> noise,
        (Tensor Key, Tensor Value)[] arPrefix, int arLength, int steps,
        Action<int, int>? onProgress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(arPrefix);
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), steps, "The solver needs at least one step.");
        if (noise.IsEmpty) throw new ArgumentException("The acoustic chunk is empty.", nameof(noise));

        float[] state = noise.ToArray();
        float[] velocity = new float[state.Length];
        float[] midpoint = new float[state.Length];
        double dt = 1.0 / steps;

        for (int step = 0; step < steps; step++)
        {
            cancel.ThrowIfCancellationRequested();
            double t = 1.0 - step * dt;

            transformer.Velocity(backend, state, LogitTimestep(t), arPrefix, arLength, velocity);
            for (int i = 0; i < state.Length; i++) midpoint[i] = state[i] - velocity[i] * (float)(dt / 2);

            cancel.ThrowIfCancellationRequested();
            transformer.Velocity(backend, midpoint, LogitTimestep(t - dt / 2), arPrefix, arLength, velocity);
            for (int i = 0; i < state.Length; i++) state[i] -= velocity[i] * (float)dt;

            onProgress?.Invoke(step + 1, steps);
        }

        for (int i = 0; i < state.Length; i++)
        {
            if (!float.IsFinite(state[i]))
                throw new InvalidOperationException("YuE2 acoustic flow matching produced non-finite latents.");
        }
        return state;
    }

    /// <summary>The solver's <c>t</c> as the model wants it. Double precision matters at the endpoints: at
    /// <c>t = 1</c> the logit is infinite and only the clamp keeps the first step finite.</summary>
    private static float LogitTimestep(double t)
        => (float)Math.Clamp(Math.Log(t / (1.0 - t)), -20.0, 20.0);
}
