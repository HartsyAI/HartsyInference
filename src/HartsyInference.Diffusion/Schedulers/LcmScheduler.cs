using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Latent Consistency Model (LCM) scheduler. Enables 1-4 step inference by treating the diffusion process as a consistency mapping rather than iterative denoising. Compatible with LCM-LoRA fine-tuned models and Hyper-SD distilled models. Uses DDPM-style noise schedule with configurable original inference steps.</summary>
public sealed unsafe class LcmScheduler : IScheduler
{
    // LCM boundary-condition constants (diffusers LCMScheduler defaults). sigma_data
    // fixes the consistency-function skip/out scaling; timestep_scaling maps the
    // discrete timestep into the continuous boundary formula.
    private const float SigmaData = 0.5f;
    private const float TimestepScaling = 10.0f;

    private readonly int _numTrainTimesteps;
    private readonly int _originalInferenceSteps;
    private readonly float _betaStart;
    private readonly float _betaEnd;
    private float[] _timesteps = [];
    private float[] _alphasCumprod = [];
    private int _numInferenceSteps;

    /// <summary>Seed for the per-step noise re-injection. Set to the generation seed before sampling for reproducibility; each step derives a distinct deterministic stream from it.</summary>
    public int Seed { get; set; }

    /// <inheritdoc/>
    public string Name => "lcm";

    /// <inheritdoc/>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <inheritdoc/>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>Initial noise sigma (always 1.0 for LCM).</summary>
    public float InitialNoiseSigma => 1.0f;

    /// <summary>Creates an LCM scheduler.</summary>
    /// <param name="numTrainTimesteps">Total timesteps in the training schedule (typically 1000).</param>
    /// <param name="betaStart">Starting beta value (0.00085 for SD).</param>
    /// <param name="betaEnd">Ending beta value (0.012 for SD).</param>
    /// <param name="originalInferenceSteps">Original inference steps the LCM was distilled from (typically 50).</param>
    public LcmScheduler(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f,
        int originalInferenceSteps = 50)
    {
        _numTrainTimesteps = numTrainTimesteps;
        _betaStart = betaStart;
        _betaEnd = betaEnd;
        _originalInferenceSteps = originalInferenceSteps;

        ComputeAlphasCumprod();
    }

    /// <summary>Sets the timestep schedule for the given number of inference steps (typically 1-4 for LCM).</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;

        // LCM uses a subset of the original training schedule
        // Evenly-spaced timesteps from the original schedule
        int stepRatio = _numTrainTimesteps / _originalInferenceSteps;
        float[] lcmOriginTimesteps = new float[_originalInferenceSteps];
        for (int i = 0; i < _originalInferenceSteps; i++)
        {
            lcmOriginTimesteps[i] = (float)((i + 1) * stepRatio - 1);
        }

        // Select the subset of timesteps for the desired inference steps
        _timesteps = new float[numInferenceSteps];
        float skippingStep = (float)_originalInferenceSteps / numInferenceSteps;
        for (int i = 0; i < numInferenceSteps; i++)
        {
            int idx = Math.Min((int)(skippingStep * (i + 1)) - 1, _originalInferenceSteps - 1);
            _timesteps[numInferenceSteps - 1 - i] = lcmOriginTimesteps[idx];
        }
    }

    /// <summary>Performs one LCM step. The model predicts epsilon; LCM recovers x0, applies the consistency boundary scaling, then for every step except the last re-noises to the next (less noisy) level with fresh Gaussian noise. The final step returns the clean estimate directly.</summary>
    public unsafe void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex)
    {
        int timestep = (int)_timesteps[stepIndex];
        bool isLast = stepIndex + 1 >= _numInferenceSteps;

        float alphaProdT = _alphasCumprod[timestep];
        float betaProdT = 1.0f - alphaProdT;
        float sqrtAlphaProdT = MathF.Sqrt(alphaProdT);
        float sqrtBetaProdT = MathF.Sqrt(betaProdT);

        // Consistency-function boundary scalings: denoised = c_out·x0 + c_skip·sample.
        float scaledTimestep = timestep * TimestepScaling;
        float denom = scaledTimestep * scaledTimestep + SigmaData * SigmaData;
        float cSkip = (SigmaData * SigmaData) / denom;
        float cOut = scaledTimestep / MathF.Sqrt(denom);

        // Re-noise coefficients for the next (less noisy) timestep, if any.
        float sqrtAlphaProdTPrev = 1.0f;
        float sqrtBetaProdTPrev = 0.0f;
        if (!isLast)
        {
            int prevTimestep = (int)_timesteps[stepIndex + 1];
            float alphaProdTPrev = _alphasCumprod[prevTimestep];
            sqrtAlphaProdTPrev = MathF.Sqrt(alphaProdTPrev);
            sqrtBetaProdTPrev = MathF.Sqrt(1.0f - alphaProdTPrev);
        }

        int count = (int)sample.ElementCount;
        // Fresh Gaussian noise for re-injection. Derive a distinct deterministic stream
        // per step so reruns with the same Seed are bit-reproducible.
        Tensor? noise = isLast ? null : SeedGenerator.CreateNoise(sample.Shape, unchecked(Seed * 9973 + stepIndex + 1));
        try
        {
            float* modelPtr = (float*)modelOutput.DataPointer;
            float* samplePtr = (float*)sample.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            float* noisePtr = noise is null ? null : (float*)noise.DataPointer;

            for (int i = 0; i < count; i++)
            {
                float predX0 = (samplePtr[i] - sqrtBetaProdT * modelPtr[i]) / sqrtAlphaProdT;
                float denoised = cOut * predX0 + cSkip * samplePtr[i];
                outPtr[i] = noisePtr is null
                    ? denoised
                    : sqrtAlphaProdTPrev * denoised + sqrtBetaProdTPrev * noisePtr[i];
            }
        }
        finally
        {
            noise?.Dispose();
        }
    }

    /// <summary>Adds noise to a clean sample (forward diffusion) for img2img starts: <c>noisy = sqrt(alpha_t)·sample + sqrt(1-alpha_t)·noise</c>.</summary>
    public unsafe void AddNoise(Tensor output, Tensor sample, Tensor noise, int stepIndex)
    {
        int timestep = (int)_timesteps[stepIndex];
        float sqrtAlphaCumprod = MathF.Sqrt(_alphasCumprod[timestep]);
        float sqrtOneMinusAlphaCumprod = MathF.Sqrt(1.0f - _alphasCumprod[timestep]);

        float* samplePtr = (float*)sample.DataPointer;
        float* noisePtr = (float*)noise.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        for (int i = 0; i < count; i++)
        {
            outPtr[i] = sqrtAlphaCumprod * samplePtr[i] + sqrtOneMinusAlphaCumprod * noisePtr[i];
        }
    }

    private void ComputeAlphasCumprod()
    {
        _alphasCumprod = new float[_numTrainTimesteps];
        float cumprod = 1.0f;

        for (int i = 0; i < _numTrainTimesteps; i++)
        {
            // Scaled linear schedule: beta = (sqrt(beta_start) + i * (sqrt(beta_end) - sqrt(beta_start)) / (T-1))^2
            float t = (float)i / (_numTrainTimesteps - 1);
            float sqrtBeta = MathF.Sqrt(_betaStart) + t * (MathF.Sqrt(_betaEnd) - MathF.Sqrt(_betaStart));
            float beta = sqrtBeta * sqrtBeta;
            float alpha = 1.0f - beta;
            cumprod *= alpha;
            _alphasCumprod[i] = cumprod;
        }
    }
}
