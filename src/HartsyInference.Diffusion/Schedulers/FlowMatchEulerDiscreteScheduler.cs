using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Flow-match Euler discrete scheduler for SD3 and Flux. Uses velocity prediction with a shifted sigma schedule. Matches HuggingFace diffusers FlowMatchEulerDiscreteScheduler.</summary>
public sealed class FlowMatchEulerDiscreteScheduler : IScheduler
{
    private readonly float _shift;
    private readonly float? _shiftTerminal;
    private float[] _sigmas;
    private float[] _timesteps;
    private int _numInferenceSteps;

    /// <summary>Name of this scheduler.</summary>
    public string Name => "flow_match_euler";

    /// <summary>Number of inference steps configured.</summary>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <summary>The computed timestep schedule (sigma values scaled to 0-1000 range).</summary>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>Returns the initial noise scale factor (sigma[0] for flow matching).</summary>
    public float InitialNoiseSigma => _sigmas.Length > 0 ? _sigmas[0] : 1.0f;

    /// <summary>Flow-match schedulers do not scale model input. Always returns 1.0.</summary>
    public float ScaleModelInput(int stepIndex) => 1.0f;

    /// <summary>Creates a flow-match Euler scheduler with the specified shift value.</summary>
    /// <param name="shift">Schedule shift parameter. 3.0 for SD3 Medium, dynamic for Flux.</param>
    /// <param name="shiftTerminal">Optional terminal sigma to stretch the schedule's last non-zero step to
    /// (diffusers <c>shift_terminal</c>). Qwen-Image sets 0.02; Flux/SD3 leave it null (no stretch).</param>
    public FlowMatchEulerDiscreteScheduler(float shift = 3.0f, float? shiftTerminal = null)
    {
        _shift = shift;
        _shiftTerminal = shiftTerminal;
        _sigmas = Array.Empty<float>();
        _timesteps = Array.Empty<float>();
    }

    /// <summary>Configures the scheduler for the given number of inference steps.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;

        // Compute sigma schedule: linearly spaced from 1.0 to 0.0 (N+1 points, includes terminal 0)
        // Then apply shift: sigma = shift * t / (1 + (shift - 1) * t)
        _sigmas = new float[numInferenceSteps + 1];
        _timesteps = new float[numInferenceSteps];

        for (int i = 0; i <= numInferenceSteps; i++)
        {
            // t goes from 1.0 to 0.0 linearly
            float t = 1.0f - (float)i / numInferenceSteps;

            // Apply shift: sigma = shift * t / (1 + (shift - 1) * t)
            float sigma = _shift * t / (1.0f + (_shift - 1.0f) * t);
            _sigmas[i] = sigma;
        }

        // Optional terminal stretch (diffusers stretch_shift_to_terminal): rescale the non-zero sigmas so the
        // last one lands exactly on shiftTerminal, keeping sigma[0]≈1. Index N stays the appended terminal 0.
        // one_minus_z = 1 - sigma; scale = one_minus_z[last] / (1 - shiftTerminal); sigma = 1 - one_minus_z/scale.
        if (_shiftTerminal is float terminal && numInferenceSteps > 0)
        {
            float oneMinusLast = 1.0f - _sigmas[numInferenceSteps - 1];
            float scale = oneMinusLast / (1.0f - terminal);
            for (int i = 0; i < numInferenceSteps; i++)
            {
                _sigmas[i] = 1.0f - (1.0f - _sigmas[i]) / scale;
            }
        }

        // Timesteps are sigmas scaled to 0-1000 range (for model conditioning)
        for (int i = 0; i < numInferenceSteps; i++)
        {
            _timesteps[i] = _sigmas[i] * 1000.0f;
        }
    }

    /// <summary>The flow-match Euler step size for a step: <c>dt = sigma_next - sigma</c> (negative). Lets callers run
    /// the step <c>x_next = x + v·dt</c> on-device (Scale+Add) instead of the host <see cref="Step"/> loop, keeping the
    /// latent GPU-resident across the sampling loop.</summary>
    public float Dt(int stepIndex) => _sigmas[stepIndex + 1] - _sigmas[stepIndex];

    /// <summary>Exact schedule sigma at an inference-step boundary, including the terminal index
    /// <c>NumInferenceSteps</c> whose value is zero. Pipeline glue uses this instead of round-tripping
    /// <see cref="Timesteps"/> through the public 0-1000 conditioning scale.
    /// <para>Public rather than internal since 2026-08-20: the sampler seam's on-schedule timestep check
    /// (<c>sigma == SigmaAt(stepIndex)</c>, which keeps a converted pipeline bit-identical on the default path) is
    /// needed by pipelines in <c>HartsyInference.Video</c>, and this assembly grants <c>InternalsVisibleTo</c> only
    /// to its own test projects.</para></summary>
    public float SigmaAt(int stepIndex)
    {
        if ((uint)stepIndex >= (uint)_sigmas.Length)
            throw new ArgumentOutOfRangeException(nameof(stepIndex), stepIndex,
                $"Sigma index must be in [0,{_sigmas.Length - 1}].");
        return _sigmas[stepIndex];
    }

    /// <summary>The full sigma array (length <c>steps + 1</c>, descending, terminal zero) — what
    /// <c>Sampling.SigmaSchedule</c> re-spaces and what an <c>ISampler</c> integrates over. A copy, so a sampler
    /// holding it cannot perturb the scheduler's own state.
    /// <para>Note that a flow-match family's conditioning timestep IS its sigma (the public <see cref="Timesteps"/>
    /// scale is just <c>sigma x 1000</c>), so a pipeline's predictor can map sigma to its timestep directly instead of
    /// indexing a table — which is what lets second-order samplers evaluate at intermediate sigmas here.</para></summary>
    public float[] Sigmas() => (float[])_sigmas.Clone();

    /// <summary>Performs one flow-match Euler step: x_next = x + model_output * (sigma_next - sigma).</summary>
    public unsafe void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];
        float dt = sigmaNext - sigma; // negative (sigma decreases)

        float* modelPtr = (float*)modelOutput.DataPointer;
        float* samplePtr = (float*)sample.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        // Velocity prediction: x_next = x + v * dt
        for (int i = 0; i < count; i++)
        {
            outPtr[i] = samplePtr[i] + modelPtr[i] * dt;
        }
    }

    /// <summary>Adds noise to a clean sample for img2img: noisy = (1 - sigma) * sample + sigma * noise.</summary>
    public unsafe void AddNoise(Tensor output, Tensor sample, Tensor noise, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float oneMinusSigma = 1.0f - sigma;

        float* samplePtr = (float*)sample.DataPointer;
        float* noisePtr = (float*)noise.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        // Flow matching forward process: x_t = (1 - sigma) * x_0 + sigma * noise
        for (int i = 0; i < count; i++)
        {
            outPtr[i] = oneMinusSigma * samplePtr[i] + sigma * noisePtr[i];
        }
    }

    /// <summary>Creates a FlowMatchEulerDiscreteScheduler with dynamic shift based on image resolution. Used by Flux models where shift varies with image size.</summary>
    /// <param name="imageSeqLen">Number of image tokens (H/patch * W/patch).</param>
    /// <param name="baseSeqLen">Base sequence length for shift calculation. Default: 256.</param>
    /// <param name="maxSeqLen">Max sequence length for shift calculation. Default: 4096.</param>
    /// <param name="baseShift">Base shift value. Default: 0.5.</param>
    /// <param name="maxShift">Max shift value. Default: 1.15.</param>
    /// <param name="shiftTerminal">Optional diffusers <c>shift_terminal</c> (Qwen-Image 0.02); null for Flux.</param>
    public static FlowMatchEulerDiscreteScheduler CreateWithDynamicShift(
        int imageSeqLen,
        int baseSeqLen = 256,
        int maxSeqLen = 4096,
        float baseShift = 0.5f,
        float maxShift = 1.15f,
        float? shiftTerminal = null)
    {
        // Linear interpolation of mu based on image sequence length
        float m = (maxShift - baseShift) / (maxSeqLen - baseSeqLen);
        float b = baseShift - m * baseSeqLen;
        float mu = m * imageSeqLen + b;

        // Flux uses exponential shift: shift = exp(mu)
        float shift = MathF.Exp(mu);

        return new FlowMatchEulerDiscreteScheduler(shift, shiftTerminal);
    }
}
