using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Euler discrete scheduler based on Katherine Crowson's k-diffusion. Matches HuggingFace diffusers EulerDiscreteScheduler.</summary>
public sealed class EulerDiscreteScheduler : IScheduler
{
    private readonly SchedulerConfig _config;
    private readonly float[] _trainSigmas;
    private float[] _sigmas;
    private float[] _timesteps;
    private int _numInferenceSteps;
    private bool _useKarrasSigmas;

    /// <inheritdoc/>
    public string Name => "euler";

    /// <inheritdoc/>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <inheritdoc/>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary><c>sigma[0]</c> for linspace/trailing spacing; <c>sqrt(sigma[0]^2 + 1)</c> otherwise, matching diffusers' variance-preserving init scale.</summary>
    public float InitialNoiseSigma
    {
        get
        {
            float maxSigma = _sigmas[0];
            if (_config.TimestepSpacing == TimestepSpacing.Linspace ||
                _config.TimestepSpacing == TimestepSpacing.Trailing)
            {
                return maxSigma;
            }
            return MathF.Sqrt(maxSigma * maxSigma + 1.0f);
        }
    }

    /// <summary>Scales the model input by 1/sqrt(sigma^2 + 1) as required by Euler schedulers.</summary>
    public float ScaleModelInput(int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        return 1.0f / MathF.Sqrt(sigma * sigma + 1.0f);
    }

    /// <summary>Creates an Euler discrete scheduler with the given configuration.</summary>
    public EulerDiscreteScheduler(SchedulerConfig? config = null, bool useKarrasSigmas = false)
    {
        _config = config ?? new SchedulerConfig();
        _useKarrasSigmas = useKarrasSigmas;
        _sigmas = Array.Empty<float>();
        _timesteps = Array.Empty<float>();

        float[] betas = NoiseSchedule.ComputeBetas(_config);
        float[] alphas = NoiseSchedule.ComputeAlphas(betas);
        float[] alphasCumprod = NoiseSchedule.ComputeAlphasCumprod(alphas);
        _trainSigmas = NoiseSchedule.ComputeSigmas(alphasCumprod);
    }

    /// <summary>ComfyUI's discrete <c>percent_to_sigma</c>: the training sigma at timestep <c>(1 − percent)·(T − 1)</c>,
    /// interpolated in log sigma.</summary>
    public double SigmaAtPercent(double percent)
    {
        if (percent <= 0.0)
        {
            return 999999999.9;
        }
        if (percent >= 1.0)
        {
            return 0.0;
        }
        double t = Math.Clamp((1.0 - percent) * (_trainSigmas.Length - 1), 0.0, _trainSigmas.Length - 1);
        int lo = (int)Math.Floor(t);
        int hi = (int)Math.Ceiling(t);
        double w = t - lo;
        return Math.Exp(((1.0 - w) * Math.Log(_trainSigmas[lo])) + (w * Math.Log(_trainSigmas[hi])));
    }

    /// <summary>Configures the scheduler for the given number of inference steps.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;
        (_sigmas, _timesteps) = Compute(numInferenceSteps);
    }

    /// <summary>The sigma array <see cref="SetTimesteps"/> would build for <paramref name="numInferenceSteps"/>, without
    /// changing this scheduler.</summary>
    public float[] SigmasFor(int numInferenceSteps) => Compute(numInferenceSteps).Sigmas;

    private (float[] Sigmas, float[] Timesteps) Compute(int numInferenceSteps)
    {
        float[] sigmas;
        float[] timesteps;
        if (_useKarrasSigmas)
        {
            // Training sigmas ascend with the timestep.
            float sigmaMin = _trainSigmas[0];
            float sigmaMax = _trainSigmas[^1];
            sigmas = NoiseSchedule.ComputeKarrasSigmas(sigmaMin, sigmaMax, numInferenceSteps);

            timesteps = new float[numInferenceSteps];
            for (int i = 0; i < numInferenceSteps; i++)
            {
                timesteps[i] = SigmaToTimestep(sigmas[i]);
            }
        }
        else
        {
            timesteps = NoiseSchedule.SelectTimesteps(_config.NumTrainTimesteps, numInferenceSteps, _config.TimestepSpacing);

            // Build sigmas from timesteps by interpolating from training sigmas
            sigmas = new float[numInferenceSteps + 1];
            for (int i = 0; i < numInferenceSteps; i++)
            {
                int lowIdx = (int)timesteps[i];
                float frac = timesteps[i] - lowIdx;

                if (lowIdx + 1 < _trainSigmas.Length)
                {
                    sigmas[i] = _trainSigmas[lowIdx] * (1.0f - frac) + _trainSigmas[lowIdx + 1] * frac;
                }
                else
                {
                    sigmas[i] = _trainSigmas[lowIdx];
                }
            }
            sigmas[numInferenceSteps] = 0.0f;
        }
        return (sigmas, timesteps);
    }

    /// <summary>Whether <see cref="Step"/> reduces to <c>sample + modelOutput·dt</c> (epsilon prediction), i.e. the denoise loop may replace the host Step loop with the in-place device <c>IBackend.CfgEulerStep</c>.</summary>
    public bool FusedEulerCompatible => _config.PredictionType == PredictionType.Epsilon;

    /// <summary>Euler dt for the fused device step: <c>sigma[i+1] − sigma[i]</c>.</summary>
    public float StepDelta(int stepIndex) => _sigmas[stepIndex + 1] - _sigmas[stepIndex];

    /// <summary>The schedule's sigma at an inference-step boundary, including the terminal index
    /// <see cref="NumInferenceSteps"/> whose value is zero. Lets the sampling layer own the sigma array without
    /// round-tripping it through the public 0-1000 <see cref="Timesteps"/> scale.</summary>
    public float Sigma(int stepIndex) => _sigmas[stepIndex];

    /// <summary>The full sigma array (length <c>steps + 1</c>, descending, terminal zero) — what
    /// <c>Sampling.SigmaSchedule</c> re-spaces and what an <c>ISampler</c> integrates over.</summary>
    public float[] Sigmas() => (float[])_sigmas.Clone();

    /// <summary>Conditioning timestep for a given noise level.
    /// <para>Returns the PRECOMPUTED <see cref="Timesteps"/> entry when <paramref name="sigma"/> is exactly the schedule
    /// sigma at <paramref name="stepIndexHint"/>, and only falls back to the log-space inverse for an off-schedule value.
    /// That distinction is load-bearing for bit-identity: on the default (non-Karras) path the sigmas are interpolated
    /// FROM the timesteps, so re-deriving a timestep from its own sigma is a lossy round trip that would perturb every
    /// existing generation. Second-order samplers evaluating at an intermediate sigma legitimately take the fallback.</para></summary>
    public float TimestepForSigma(float sigma, int stepIndexHint)
    {
        if (stepIndexHint >= 0 && stepIndexHint < _timesteps.Length && _sigmas[stepIndexHint] == sigma)
        {
            return _timesteps[stepIndexHint];
        }
        return SigmaToTimestep(sigma);
    }

    /// <summary>Performs one Euler denoising step.</summary>
    public unsafe void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];

        float* modelPtr = (float*)modelOutput.DataPointer;
        float* samplePtr = (float*)sample.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        if (_config.PredictionType == PredictionType.Epsilon)
        {
            // For epsilon prediction, derivative simplifies to model_output:
            // pred_x0 = sample - sigma * eps  →  derivative = (sample - pred_x0) / sigma = eps
            // prev_sample = sample + eps * (sigma_next - sigma)
            float dt = sigmaNext - sigma;
            for (int i = 0; i < count; i++)
            {
                outPtr[i] = samplePtr[i] + modelPtr[i] * dt;
            }
        }
        else if (_config.PredictionType == PredictionType.VPrediction)
        {
            float sigmaSq = sigma * sigma;
            float sigmaSqPlus1 = sigmaSq + 1.0f;
            float sqrtSigmaSqPlus1 = MathF.Sqrt(sigmaSqPlus1);
            float dt = sigmaNext - sigma;

            if (sigma < 1e-8f)
            {
                // At sigma≈0 the sample is already fully denoised
                for (int i = 0; i < count; i++)
                {
                    outPtr[i] = samplePtr[i];
                }
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    float predX0 = modelPtr[i] * (-sigma / sqrtSigmaSqPlus1) + (samplePtr[i] / sigmaSqPlus1);
                    float derivative = (samplePtr[i] - predX0) / sigma;
                    outPtr[i] = samplePtr[i] + derivative * dt;
                }
            }
        }
    }

    /// <summary>Adds noise to a clean sample for img2img.</summary>
    public unsafe void AddNoise(Tensor output, Tensor sample, Tensor noise, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float* samplePtr = (float*)sample.DataPointer;
        float* noisePtr = (float*)noise.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        // Euler scheduler add_noise: noisy = sample + noise * sigma
        for (int i = 0; i < count; i++)
        {
            outPtr[i] = samplePtr[i] + noisePtr[i] * sigma;
        }
    }

    /// <summary>Converts a sigma value to a continuous timestep by log-linear interpolation in the training sigma schedule.</summary>
    private float SigmaToTimestep(float sigma)
    {
        // Training sigmas ascend with the timestep; interpolate the timestep in log sigma.
        if (sigma <= _trainSigmas[0])
        {
            return 0.0f;
        }
        if (sigma >= _trainSigmas[^1])
        {
            return _trainSigmas.Length - 1;
        }
        float logSigma = MathF.Log(sigma);
        int lo = 0;
        int hi = _trainSigmas.Length - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (_trainSigmas[mid] <= sigma)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        float logLo = MathF.Log(_trainSigmas[lo]);
        float logHi = MathF.Log(_trainSigmas[hi]);
        return lo + ((logSigma - logLo) / (logHi - logLo));
    }
}
