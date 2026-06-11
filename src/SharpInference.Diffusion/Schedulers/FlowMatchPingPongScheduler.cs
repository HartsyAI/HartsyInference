using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Schedulers;

/// <summary>Flow-match "ping-pong" sampler (ACE-Step's <c>FlowMatchPingPongScheduler</c>): the same shifted sigma grid
/// as <see cref="FlowMatchEulerDiscreteScheduler"/>, but each step jumps to the clean estimate and re-noises with
/// <b>fresh</b> Gaussian noise at the next level — <c>x ← (1−σ_next)·x̂₀ + σ_next·ε</c> with <c>x̂₀ = x − σ·v</c>.
/// Adds per-step stochasticity (preferred for "variations" workflows) at the cost of determinism along the ODE path.</summary>
public sealed unsafe class FlowMatchPingPongScheduler
{
    private readonly float _shift;
    private float[] _sigmas = [];
    private float[] _timesteps = [];

    public FlowMatchPingPongScheduler(float shift = 3.0f) => _shift = shift;

    /// <summary>Model-conditioning timesteps (σ·1000), one per step.</summary>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>The sigma grid (length steps + 1, terminal 0).</summary>
    public ReadOnlySpan<float> Sigmas => _sigmas;

    /// <summary>Builds the SD3-style shifted schedule: <c>σ = shift·t / (1 + (shift−1)·t)</c> over linspace t = 1 → 0.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _sigmas = new float[numInferenceSteps + 1];
        _timesteps = new float[numInferenceSteps];
        for (int i = 0; i <= numInferenceSteps; i++)
        {
            float t = 1.0f - (float)i / numInferenceSteps;
            _sigmas[i] = _shift * t / (1.0f + (_shift - 1.0f) * t);
        }
        for (int i = 0; i < numInferenceSteps; i++) _timesteps[i] = _sigmas[i] * 1000.0f;
    }

    /// <summary>One ping-pong step in place: clean estimate from the velocity, then fresh-noise re-injection at the
    /// next sigma (the terminal step lands on x̂₀ exactly since σ_next = 0).</summary>
    public void Step(Tensor sample, Tensor velocity, Tensor freshNoise, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];
        long n = sample.Shape.ElementCount;
        float* sp = (float*)sample.DataPointer;
        float* vp = (float*)velocity.DataPointer;
        float* np = (float*)freshNoise.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float x0 = sp[i] - sigma * vp[i];
            sp[i] = (1f - sigmaNext) * x0 + sigmaNext * np[i];
        }
    }
}
