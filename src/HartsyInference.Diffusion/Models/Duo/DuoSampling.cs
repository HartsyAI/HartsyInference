namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Predicts the video and audio velocities for the current state at flow time <paramref name="time"/>. The model owns any cache;
/// the sampling loop only sees the state and the time.</summary>
public delegate (float[] Video, float[] Audio) DuoVelocityModel(float[] video, float[] audio, float time);

/// <summary>The two fixed-step solvers of the Duo flow. <see cref="Euler"/> is the causal student's; <see cref="ConsistencyRenoise"/> predicts
/// the clean latent at each step and re-noises it to the next time. Upstream <c>sample_sequence</c> (<c>duo/models/realization/pipelines/inference.py</c>
/// @068e357) selects them by stage.</summary>
public enum DuoSolver
{
    Euler,
    ConsistencyRenoise,
}

/// <summary>The fixed-NFE sampling loop of the Duo flow, on CPU F32 arrays. The flow times come from
/// <see cref="HartsyInference.ModelAssets.Duo.DuoSchedule.Grid"/>. The noise used by <see cref="DuoSolver.ConsistencyRenoise"/> comes from the
/// caller, one array per step and stream, so a run can replay upstream's noise exactly; parity needs that.</summary>
public static class DuoSampling
{
    /// <summary>Runs <paramref name="steps"/> steps from the initial noise <paramref name="initialVideo"/> and <paramref name="initialAudio"/>
    /// (both at flow time 0). <paramref name="renoise"/> is called as <c>renoise(step, stream)</c> for steps before the last, with stream 0 for
    /// video and 1 for audio; it must return a noise array of that stream's length. It is only used by <see cref="DuoSolver.ConsistencyRenoise"/>.</summary>
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

        float[] schedule = HartsyInference.ModelAssets.Duo.DuoSchedule.Grid(steps, shift);
        float[] video = initialVideo, audio = initialAudio;
        for (int index = 0; index < steps; index++)
        {
            float t = schedule[index];
            (float[] predVideo, float[] predAudio) = model(video, audio, t);
            if (!IsFinite(predVideo) || !IsFinite(predAudio))
            {
                throw new InvalidOperationException($"Nonfinite Duo prediction at sampling step {index}.");
            }

            float[] nextVideo = new float[video.Length], nextAudio = new float[audio.Length];
            if (solver == DuoSolver.Euler)
            {
                DuoSampler.Euler(video, predVideo, t, schedule[index + 1], nextVideo);
                DuoSampler.Euler(audio, predAudio, t, schedule[index + 1], nextAudio);
            }
            else
            {
                DuoSampler.ToX0(video, predVideo, t, nextVideo);
                DuoSampler.ToX0(audio, predAudio, t, nextAudio);
                if (index + 1 < steps)
                {
                    float[] noiseVideo = renoise!(index, 0), noiseAudio = renoise(index, 1);
                    float[] renoisedVideo = new float[nextVideo.Length], renoisedAudio = new float[nextAudio.Length];
                    DuoSampler.AddNoise(nextVideo, noiseVideo, schedule[index + 1], renoisedVideo);
                    DuoSampler.AddNoise(nextAudio, noiseAudio, schedule[index + 1], renoisedAudio);
                    nextVideo = renoisedVideo;
                    nextAudio = renoisedAudio;
                }
            }
            video = nextVideo;
            audio = nextAudio;
        }

        if (!IsFinite(video) || !IsFinite(audio))
        {
            throw new InvalidOperationException("Nonfinite generated Duo latent.");
        }
        return (video, audio);
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
