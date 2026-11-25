using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Predicts the video and audio velocities for the current state at flow time <paramref name="time"/>. The model owns any cache.
/// It must not keep the arrays it is handed: the sampling loop overwrites them on the next step. Both solvers consume the same velocity:
/// Euler steps by it, and the consistency solver reads it as the clean-latent prediction through <c>to_x0</c> before it re-noises.</summary>
public delegate (float[] Video, float[] Audio) DuoVelocityModel(float[] video, float[] audio, float time);

/// <summary>The two fixed-step solvers of the Duo flow. <see cref="Euler"/> is the causal student's; <see cref="ConsistencyRenoise"/> predicts
/// the clean latent at each step and re-noises it to the next time. Upstream <c>sample_sequence</c> (<c>duo/models/realization/pipelines/inference.py</c>
/// @068e357) selects them by stage.</summary>
public enum DuoSolver
{
    Euler,
    ConsistencyRenoise,
}

/// <summary>The fixed-NFE sampling loop of the Duo flow, on managed CPU F32 arrays. This is the CPU reference, not the production hot path:
/// it allocates each step's predictions and runs element-wise on the host. The flow times come from <see cref="DuoSchedule.Grid"/>. The noise
/// used by <see cref="DuoSolver.ConsistencyRenoise"/> comes from the caller, one array per step and stream, so a run can replay upstream's noise
/// exactly; parity needs that.</summary>
public static class DuoSampling
{
    /// <summary>Runs <paramref name="steps"/> steps from the initial noise <paramref name="initialVideo"/> and <paramref name="initialAudio"/>
    /// (both at flow time 0). The caller's arrays are copied once and never written. <paramref name="renoise"/> is called as
    /// <c>renoise(step, stream)</c>, with stream 0 for video and 1 for audio, for every step before the last, in the order
    /// (0,0), (0,1), (1,0), (1,1), ... It must return noise the length of that stream. It is used only by <see cref="DuoSolver.ConsistencyRenoise"/>.</summary>
    /// <exception cref="ArgumentException">A velocity or a renoise array does not match its stream's length.</exception>
    /// <exception cref="InvalidOperationException">A prediction or the final state is not finite, as upstream's <c>FloatingPointError</c>.</exception>
    public static (float[] Video, float[] Audio) Sample(float[] initialVideo, float[] initialAudio, int steps, double shift,
        DuoSolver solver, DuoVelocityModel model, Func<int, int, float[]>? renoise = null)
    {
        ArgumentNullException.ThrowIfNull(initialVideo);
        ArgumentNullException.ThrowIfNull(initialAudio);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        if (solver == DuoSolver.ConsistencyRenoise && steps > 1)
        {
            ArgumentNullException.ThrowIfNull(renoise);
        }

        float[] schedule = DuoSchedule.Grid(steps, shift);
        // The loop works in place on these two buffers, so the sampler allocates nothing per step.
        float[] video = (float[])initialVideo.Clone(), audio = (float[])initialAudio.Clone();
        for (int index = 0; index < steps; index++)
        {
            float t = schedule[index];
            (float[] predVideo, float[] predAudio) = model(video, audio, t);
            CheckLength(predVideo, video.Length, $"velocity video at step {index}");
            CheckLength(predAudio, audio.Length, $"velocity audio at step {index}");
            if (!IsFinite(predVideo) || !IsFinite(predAudio))
            {
                throw new InvalidOperationException($"Nonfinite Duo prediction at sampling step {index}.");
            }

            if (solver == DuoSolver.Euler)
            {
                DuoSampler.Euler(video, predVideo, t, schedule[index + 1], video);
                DuoSampler.Euler(audio, predAudio, t, schedule[index + 1], audio);
            }
            else
            {
                DuoSampler.ToX0(video, predVideo, t, video);
                DuoSampler.ToX0(audio, predAudio, t, audio);
                if (index + 1 < steps)
                {
                    float[] noiseVideo = Noise(renoise!, index, 0, video.Length);
                    float[] noiseAudio = Noise(renoise!, index, 1, audio.Length);
                    DuoSampler.AddNoise(video, noiseVideo, schedule[index + 1], video);
                    DuoSampler.AddNoise(audio, noiseAudio, schedule[index + 1], audio);
                }
            }
        }

        if (!IsFinite(video) || !IsFinite(audio))
        {
            throw new InvalidOperationException("Nonfinite generated Duo latent.");
        }
        return (video, audio);
    }

    /// <summary>Calls the noise source for one step and stream, and checks the length at the call site.</summary>
    private static float[] Noise(Func<int, int, float[]> renoise, int step, int stream, int length)
    {
        float[] noise = renoise(step, stream) ?? throw new ArgumentNullException(nameof(renoise), $"renoise({step}, {stream}) returned null.");
        CheckLength(noise, length, $"renoise({step}, {stream})");
        return noise;
    }

    private static void CheckLength(float[] values, int expected, string what)
    {
        if (values.Length != expected)
        {
            throw new ArgumentException($"{what} has {values.Length} values; the stream has {expected}.");
        }
    }

    private static bool IsFinite(float[] values)
    {
        foreach (float value in values)
        {
            if (!float.IsFinite(value))
            {
                return false;
            }
        }
        return true;
    }
}
