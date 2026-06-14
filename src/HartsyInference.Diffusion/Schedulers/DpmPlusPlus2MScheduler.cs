using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>DPM-Solver++ 2M (second-order multistep) scheduler. Matches HuggingFace diffusers DPMSolverMultistepScheduler with solver_order=2. Lu et al. 2022.</summary>
public sealed class DpmPlusPlus2MScheduler : IScheduler
{
    private readonly SchedulerConfig _config;
    private readonly float[] _alphasCumprod;
    private float[] _sigmas;
    private float[] _timesteps;
    private int _numInferenceSteps;

    // Previous model output for multistep (stored as data prediction / x0)
    private Tensor? _prevModelOutput;
    private int _prevStepIndex;

    /// <summary>Name of this scheduler.</summary>
    public string Name => "dpm++2m";

    /// <summary>Number of inference steps configured.</summary>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <summary>The computed timestep schedule.</summary>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>Initial noise sigma (always 1.0 for DPM++).</summary>
    public float InitialNoiseSigma => 1.0f;

    /// <summary>Creates a DPM++ 2M scheduler.</summary>
    public DpmPlusPlus2MScheduler(SchedulerConfig? config = null)
    {
        _config = config ?? new SchedulerConfig();
        _sigmas = Array.Empty<float>();
        _timesteps = Array.Empty<float>();

        float[] betas = NoiseSchedule.ComputeBetas(_config);
        float[] alphas = NoiseSchedule.ComputeAlphas(betas);
        _alphasCumprod = NoiseSchedule.ComputeAlphasCumprod(alphas);
    }

    /// <summary>Configures the scheduler for the given number of inference steps.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;
        _timesteps = NoiseSchedule.SelectTimesteps(_config.NumTrainTimesteps, numInferenceSteps, _config.TimestepSpacing);

        // Compute sigmas from timesteps
        float[] trainSigmas = NoiseSchedule.ComputeSigmas(_alphasCumprod);
        _sigmas = new float[numInferenceSteps + 1];
        for (int i = 0; i < numInferenceSteps; i++)
        {
            int idx = (int)_timesteps[i];
            _sigmas[i] = idx < trainSigmas.Length ? trainSigmas[idx] : trainSigmas[^1];
        }
        _sigmas[numInferenceSteps] = 0.0f;

        // Reset multistep state
        _prevModelOutput?.Dispose();
        _prevModelOutput = null;
        _prevStepIndex = -1;
    }

    /// <summary>Performs one DPM++ 2M denoising step.</summary>
    public unsafe void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex)
    {
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];

        // Convert sigma to alpha_t, sigma_t (continuous form)
        SigmaToAlphaSigma(sigma, out float alphaS, out float sigmaS);
        SigmaToAlphaSigma(sigmaNext, out float alphaT, out float sigmaT);

        // Convert model output to data prediction (predicted x0)
        Tensor x0Pred = ConvertModelOutput(modelOutput, sample, sigma, alphaS, sigmaS);

        try
        {
            if (_prevModelOutput is null || _prevStepIndex != stepIndex - 1)
            {
                // First order update (no previous output available)
                FirstOrderUpdate(output, x0Pred, sample, alphaS, sigmaS, alphaT, sigmaT);
            }
            else
            {
                // Second order multistep update
                float sigmaPrev = _sigmas[_prevStepIndex];
                SigmaToAlphaSigma(sigmaPrev, out float alphaS1, out float sigmaS1);

                SecondOrderUpdate(output, x0Pred, _prevModelOutput, sample,
                    alphaS, sigmaS, alphaS1, sigmaS1, alphaT, sigmaT);
            }

            // Store current x0 prediction for next multistep
            _prevModelOutput?.Dispose();
            _prevModelOutput = new Tensor(x0Pred.Shape, x0Pred.DType, x0Pred.Device);
            long byteSize = Tensor.ComputeByteSize(x0Pred.Shape, x0Pred.DType);
            Buffer.MemoryCopy(x0Pred.DataPointer, _prevModelOutput.DataPointer, byteSize, byteSize);
            _prevStepIndex = stepIndex;
        }
        finally
        {
            x0Pred.Dispose();
        }
    }

    /// <summary>Adds noise to a clean sample (forward diffusion process).</summary>
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

    /// <summary>Converts sigma to (alpha_t, sigma_t) pair in continuous form.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SigmaToAlphaSigma(float sigma, out float alphaT, out float sigmaT)
    {
        alphaT = 1.0f / MathF.Sqrt(sigma * sigma + 1.0f);
        sigmaT = sigma * alphaT;
    }

    /// <summary>Converts model output to data prediction (predicted x0).</summary>
    private unsafe Tensor ConvertModelOutput(Tensor modelOutput, Tensor sample, float sigma, float alphaT, float sigmaT)
    {
        Tensor x0Pred = new Tensor(sample.Shape, sample.DType, sample.Device);
        float* modelPtr = (float*)modelOutput.DataPointer;
        float* samplePtr = (float*)sample.DataPointer;
        float* x0Ptr = (float*)x0Pred.DataPointer;
        int count = (int)sample.ElementCount;

        if (_config.PredictionType == PredictionType.Epsilon)
        {
            // x0 = (sample - sigma_t * eps) / alpha_t
            for (int i = 0; i < count; i++)
            {
                x0Ptr[i] = (samplePtr[i] - sigmaT * modelPtr[i]) / alphaT;
            }
        }
        else if (_config.PredictionType == PredictionType.VPrediction)
        {
            // x0 = alpha_t * sample - sigma_t * v
            for (int i = 0; i < count; i++)
            {
                x0Ptr[i] = alphaT * samplePtr[i] - sigmaT * modelPtr[i];
            }
        }

        return x0Pred;
    }

    /// <summary>DPM-Solver++ first-order update.</summary>
    private unsafe void FirstOrderUpdate(Tensor output, Tensor x0Pred, Tensor sample,
        float alphaS, float sigmaS, float alphaT, float sigmaT)
    {
        float lambdaT = MathF.Log(alphaT) - MathF.Log(sigmaT);
        float lambdaS = MathF.Log(alphaS) - MathF.Log(sigmaS);
        float h = lambdaT - lambdaS;
        float coeffSample = sigmaT / sigmaS;
        float coeffX0 = alphaT * (MathF.Exp(-h) - 1.0f);

        float* x0Ptr = (float*)x0Pred.DataPointer;
        float* samplePtr = (float*)sample.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        for (int i = 0; i < count; i++)
        {
            outPtr[i] = coeffSample * samplePtr[i] - coeffX0 * x0Ptr[i];
        }
    }

    /// <summary>DPM-Solver++ second-order multistep update (2M midpoint).</summary>
    private unsafe void SecondOrderUpdate(Tensor output, Tensor m0, Tensor m1, Tensor sample,
        float alphaS0, float sigmaS0, float alphaS1, float sigmaS1, float alphaT, float sigmaT)
    {
        float lambdaT = MathF.Log(alphaT) - MathF.Log(sigmaT);
        float lambdaS0 = MathF.Log(alphaS0) - MathF.Log(sigmaS0);
        float lambdaS1 = MathF.Log(alphaS1) - MathF.Log(sigmaS1);

        float h = lambdaT - lambdaS0;
        float h0 = lambdaS0 - lambdaS1;
        float r0 = h0 / h;

        float coeffSample = sigmaT / sigmaS0;
        float expNegH = MathF.Exp(-h);
        float coeffD0 = alphaT * (expNegH - 1.0f);
        float coeffD1 = 0.5f * alphaT * (expNegH - 1.0f);

        float* m0Ptr = (float*)m0.DataPointer;
        float* m1Ptr = (float*)m1.DataPointer;
        float* samplePtr = (float*)sample.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        for (int i = 0; i < count; i++)
        {
            float D0 = m0Ptr[i];
            float D1 = (1.0f / r0) * (m0Ptr[i] - m1Ptr[i]);
            outPtr[i] = coeffSample * samplePtr[i] - coeffD0 * D0 - coeffD1 * D1;
        }
    }
}
