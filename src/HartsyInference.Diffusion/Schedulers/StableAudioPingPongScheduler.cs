using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Stable Audio Open Small's "pingpong" rectified-flow sampler (<c>stable_audio_tools.inference.
/// sampling.sample_rf(sampler_type="pingpong")</c> → <c>sample_flow_pingpong</c>) — 8 steps, no CFG
/// (ARC-distilled). Uses a logSNR-linspace sigma grid, NOT the SD3-shift grid <see cref="FlowMatchPingPongScheduler"/>
/// builds for ACE-Step (a different, non-transferable schedule — kept as a separate class rather than
/// parameterizing that one). Each step jumps to the clean estimate then re-noises with fresh Gaussian noise at
/// the next sigma level; the terminal step lands exactly on the clean estimate since <c>t[steps] = 0</c>.</summary>
public sealed unsafe class StableAudioPingPongScheduler
{
    private float[] _sigmas = [];

    /// <summary>The sigma/time grid (length steps + 1; <c>t[0] = sigmaMax</c>, <c>t[steps] = 0</c>). Fed directly
    /// as <c>StableAudioDit.Forward</c>'s <c>timestep</c> argument — no ×1000 scaling (unlike ACE-Step).</summary>
    public ReadOnlySpan<float> Sigmas => _sigmas;

    /// <summary>Builds <c>sample_rf</c>'s schedule: <c>logsnr = linspace(logsnrMax, 2, steps+1)</c>,
    /// <c>t = sigmoid(-logsnr)</c>, endpoints pinned to <paramref name="sigmaMax"/> and 0.</summary>
    public void SetTimesteps(int numInferenceSteps, float sigmaMax = 1f)
    {
        float logsnrMax = sigmaMax < 1f ? MathF.Log((1f - sigmaMax) / sigmaMax + 1e-6f) : -6f;
        _sigmas = new float[numInferenceSteps + 1];
        for (int i = 0; i <= numInferenceSteps; i++)
        {
            float logsnr = logsnrMax + i * (2f - logsnrMax) / numInferenceSteps;
            _sigmas[i] = 1f / (1f + MathF.Exp(logsnr));
        }
        _sigmas[0] = sigmaMax;
        _sigmas[numInferenceSteps] = 0f;
    }

    /// <summary>One ping-pong step in place: <c>x̂₀ = x − t·v</c>, then fresh-noise re-injection at the next t
    /// (<c>x ← (1 − t_next)·x̂₀ + t_next·ε</c>).</summary>
    public void Step(Tensor sample, Tensor velocity, Tensor freshNoise, int stepIndex)
    {
        float t = _sigmas[stepIndex];
        float tNext = _sigmas[stepIndex + 1];
        long n = sample.Shape.ElementCount;
        float* sp = (float*)sample.DataPointer;
        float* vp = (float*)velocity.DataPointer;
        float* np = (float*)freshNoise.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float x0 = sp[i] - t * vp[i];
            sp[i] = (1f - tNext) * x0 + tNext * np[i];
        }
    }
}
