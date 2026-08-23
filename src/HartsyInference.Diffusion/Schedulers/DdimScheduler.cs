using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>DDIM (Denoising Diffusion Implicit Models) scheduler. Matches HuggingFace diffusers DDIMScheduler. Song et al. 2020.</summary>
public sealed class DdimScheduler : IScheduler
{
    private readonly SchedulerConfig _config;
    private readonly float[] _alphasCumprod;
    private readonly float _finalAlphaCumprod;
    private float[] _timesteps;
    private int _numInferenceSteps;
    private float _eta;

    /// <inheritdoc/>
    public string Name => "ddim";

    /// <inheritdoc/>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <inheritdoc/>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>Initial noise sigma (always 1.0 for DDIM).</summary>
    public float InitialNoiseSigma => 1.0f;

    /// <summary>Creates a DDIM scheduler.</summary>
    /// <param name="config">Scheduler configuration.</param>
    /// <param name="eta">DDIM eta parameter. 0.0 = deterministic DDIM, 1.0 = DDPM equivalent.</param>
    /// <param name="setAlphaToOne">If true, final_alpha_cumprod = 1.0; if false, uses alphas_cumprod[0].</param>
    public DdimScheduler(SchedulerConfig? config = null, float eta = 0.0f, bool setAlphaToOne = true)
    {
        _config = config ?? new SchedulerConfig();
        _eta = eta;
        _timesteps = Array.Empty<float>();

        float[] betas = NoiseSchedule.ComputeBetas(_config);
        float[] alphas = NoiseSchedule.ComputeAlphas(betas);
        _alphasCumprod = NoiseSchedule.ComputeAlphasCumprod(alphas);
        _finalAlphaCumprod = setAlphaToOne ? 1.0f : _alphasCumprod[0];
    }

    /// <summary>Configures the scheduler for the given number of inference steps.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;
        _timesteps = NoiseSchedule.SelectTimesteps(_config.NumTrainTimesteps, numInferenceSteps, _config.TimestepSpacing);
    }

    /// <summary>Performs one DDIM denoising step.</summary>
    public unsafe void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex)
    {
        int timestep = (int)_timesteps[stepIndex];
        int prevTimestep = stepIndex + 1 < _numInferenceSteps
            ? (int)_timesteps[stepIndex + 1]
            : -1;

        float alphaProdT = _alphasCumprod[timestep];
        float alphaProdTPrev = prevTimestep >= 0 ? _alphasCumprod[prevTimestep] : _finalAlphaCumprod;
        float betaProdT = 1.0f - alphaProdT;
        float betaProdTPrev = 1.0f - alphaProdTPrev;

        float sqrtAlphaProdT = MathF.Sqrt(alphaProdT);
        float sqrtBetaProdT = MathF.Sqrt(betaProdT);
        float sqrtAlphaProdTPrev = MathF.Sqrt(alphaProdTPrev);

        // Compute variance and std_dev_t
        float variance = (betaProdTPrev / betaProdT) * (1.0f - alphaProdT / alphaProdTPrev);
        float stdDevT = _eta * MathF.Sqrt(variance);

        // Direction coefficient: sqrt(1 - alpha_prev - sigma_t^2)
        float directionCoeff = MathF.Sqrt(MathF.Max(0.0f, 1.0f - alphaProdTPrev - stdDevT * stdDevT));

        float* modelPtr = (float*)modelOutput.DataPointer;
        float* samplePtr = (float*)sample.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        if (_config.PredictionType == PredictionType.Epsilon)
        {
            for (int i = 0; i < count; i++)
            {
                // pred_x0 = (sample - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
                float predX0 = (samplePtr[i] - sqrtBetaProdT * modelPtr[i]) / sqrtAlphaProdT;
                // pred_sample_direction = sqrt(1 - alpha_prev - sigma_t^2) * eps
                float predDirection = directionCoeff * modelPtr[i];
                // prev_sample = sqrt(alpha_prev) * pred_x0 + pred_direction
                outPtr[i] = sqrtAlphaProdTPrev * predX0 + predDirection;
            }
        }
        else if (_config.PredictionType == PredictionType.VPrediction)
        {
            for (int i = 0; i < count; i++)
            {
                // pred_x0 = sqrt(alpha_t) * sample - sqrt(1-alpha_t) * model_output
                float predX0 = sqrtAlphaProdT * samplePtr[i] - sqrtBetaProdT * modelPtr[i];
                // pred_eps = sqrt(alpha_t) * model_output + sqrt(1-alpha_t) * sample
                float predEps = sqrtAlphaProdT * modelPtr[i] + sqrtBetaProdT * samplePtr[i];
                float predDirection = directionCoeff * predEps;
                outPtr[i] = sqrtAlphaProdTPrev * predX0 + predDirection;
            }
        }

        // Note: stochastic noise (eta > 0) would be added by the caller providing a noise tensor.
        // For deterministic DDIM (eta=0), stdDevT=0 and no noise is needed.
    }

    /// <summary>Adds noise to a clean sample (forward diffusion process).</summary>
    public void AddNoise(Tensor output, Tensor sample, Tensor noise, int stepIndex)
        => NoiseSchedule.AddNoise(output, sample, noise, _timesteps, _alphasCumprod, stepIndex);
}
