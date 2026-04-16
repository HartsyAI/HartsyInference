using SharpInference.Core.Schedulers;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Schedulers;

/// <summary>Euler discrete scheduler based on Katherine Crowson's k-diffusion. Matches HuggingFace diffusers EulerDiscreteScheduler.</summary>
public sealed class EulerDiscreteScheduler : IScheduler
{
    private readonly SchedulerConfig _config;
    private readonly float[] _trainSigmas;
    private float[] _sigmas;
    private float[] _timesteps;
    private int _numInferenceSteps;
    private bool _useKarrasSigmas;

    /// <summary>Name of this scheduler.</summary>
    public string Name => "euler";

    /// <summary>Number of inference steps configured.</summary>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <summary>The computed timestep schedule.</summary>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>Returns the initial noise scale factor.</summary>
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

    /// <summary>Configures the scheduler for the given number of inference steps.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;

        if (_useKarrasSigmas)
        {
            float sigmaMin = _trainSigmas[^1] > 0 ? _trainSigmas[^1] : _trainSigmas[^2];
            float sigmaMax = _trainSigmas[0];
            _sigmas = NoiseSchedule.ComputeKarrasSigmas(sigmaMin, sigmaMax, numInferenceSteps);

            _timesteps = new float[numInferenceSteps];
            for (int i = 0; i < numInferenceSteps; i++)
            {
                _timesteps[i] = SigmaToTimestep(_sigmas[i]);
            }
        }
        else
        {
            _timesteps = NoiseSchedule.SelectTimesteps(_config.NumTrainTimesteps, numInferenceSteps, _config.TimestepSpacing);

            // Build sigmas from timesteps by interpolating from training sigmas
            _sigmas = new float[numInferenceSteps + 1];
            for (int i = 0; i < numInferenceSteps; i++)
            {
                int lowIdx = (int)_timesteps[i];
                float frac = _timesteps[i] - lowIdx;

                if (lowIdx + 1 < _trainSigmas.Length)
                {
                    _sigmas[i] = _trainSigmas[lowIdx] * (1.0f - frac) + _trainSigmas[lowIdx + 1] * frac;
                }
                else
                {
                    _sigmas[i] = _trainSigmas[lowIdx];
                }
            }
            _sigmas[numInferenceSteps] = 0.0f;
        }
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
            // pred_x0 = sample - sigma * model_output
            // derivative = (sample - pred_x0) / sigma = model_output
            // prev_sample = sample + derivative * (sigma_next - sigma)
            float dt = sigmaNext - sigma;
            for (int i = 0; i < count; i++)
            {
                float predX0 = samplePtr[i] - sigma * modelPtr[i];
                float derivative = (samplePtr[i] - predX0) / sigma;
                outPtr[i] = samplePtr[i] + derivative * dt;
            }
        }
        else if (_config.PredictionType == PredictionType.VPrediction)
        {
            float sigmaSq = sigma * sigma;
            float sigmaSqPlus1 = sigmaSq + 1.0f;
            float sqrtSigmaSqPlus1 = MathF.Sqrt(sigmaSqPlus1);
            float dt = sigmaNext - sigma;

            for (int i = 0; i < count; i++)
            {
                float predX0 = modelPtr[i] * (-sigma / sqrtSigmaSqPlus1) + (samplePtr[i] / sigmaSqPlus1);
                float derivative = (samplePtr[i] - predX0) / sigma;
                outPtr[i] = samplePtr[i] + derivative * dt;
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
        float logSigma = MathF.Log(sigma);

        // Find where logSigma falls in the training log-sigma schedule (reversed: high to low)
        for (int i = 0; i < _trainSigmas.Length - 1; i++)
        {
            float logLow = MathF.Log(_trainSigmas[i]);
            float logHigh = MathF.Log(_trainSigmas[i + 1]);

            if (logSigma >= logHigh && logSigma <= logLow)
            {
                float w = (logSigma - logHigh) / (logLow - logHigh);
                return (1.0f - w) * (i + 1) + w * i;
            }
        }

        // Clamp to boundaries
        if (sigma >= _trainSigmas[0]) return 0.0f;
        return _trainSigmas.Length - 1;
    }
}
