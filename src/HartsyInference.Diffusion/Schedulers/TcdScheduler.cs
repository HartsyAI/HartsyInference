using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Trajectory Consistency Distillation (TCD) scheduler. Like <see cref="LcmScheduler"/> it uses the consistency boundary mapping, but instead of re-noising fully to the next timestep it jumps to an intermediate state <c>s = floor((1-eta)·t_prev)</c> controlled by <see cref="Eta"/>, then optionally adds stochasticity. <c>Eta=0</c> is deterministic (DDIM-like on the consistency estimate); <c>Eta=1</c> reduces to LCM full re-noise. The tunable fraction is why TCD holds quality across 4-50 steps where LCM degrades past ~8. Compatible with TCD-LoRA and the same distilled checkpoints LCM-LoRA targets.</summary>
public sealed unsafe class TcdScheduler : IScheduler
{
    private const float SigmaData = 0.5f;
    private const float TimestepScaling = 10.0f;

    private readonly int _numTrainTimesteps;
    private readonly int _originalInferenceSteps;
    private readonly float _betaStart;
    private readonly float _betaEnd;
    private float[] _timesteps = [];
    private float[] _alphasCumprod = [];
    private int _numInferenceSteps;

    /// <summary>Stochasticity parameter (gamma) in [0, 1]. 0 = deterministic, 1 = LCM-equivalent full re-noise. The TCD paper's default sweet spot is ~0.3.</summary>
    public float Eta { get; set; } = 0.3f;

    /// <summary>Seed for per-step noise; set to the generation seed for reproducibility.</summary>
    public int Seed { get; set; }

    /// <inheritdoc/>
    public string Name => "tcd";

    /// <inheritdoc/>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <inheritdoc/>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>Initial noise sigma (always 1.0 for TCD).</summary>
    public float InitialNoiseSigma => 1.0f;

    /// <summary>Creates a TCD scheduler.</summary>
    /// <param name="numTrainTimesteps">Total timesteps in the training schedule (typically 1000).</param>
    /// <param name="betaStart">Starting beta value (0.00085 for SD).</param>
    /// <param name="betaEnd">Ending beta value (0.012 for SD).</param>
    /// <param name="originalInferenceSteps">Original inference steps the model was distilled from (typically 50).</param>
    public TcdScheduler(int numTrainTimesteps = 1000, float betaStart = 0.00085f, float betaEnd = 0.012f,
        int originalInferenceSteps = 50)
    {
        _numTrainTimesteps = numTrainTimesteps;
        _betaStart = betaStart;
        _betaEnd = betaEnd;
        _originalInferenceSteps = originalInferenceSteps;
        ComputeAlphasCumprod();
    }

    /// <summary>Sets the timestep schedule (descending) for the given number of inference steps, selecting a subset of the distilled origin schedule.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;
        int stepRatio = _numTrainTimesteps / _originalInferenceSteps;
        float[] originTimesteps = new float[_originalInferenceSteps];
        for (int i = 0; i < _originalInferenceSteps; i++)
        {
            originTimesteps[i] = (i + 1) * stepRatio - 1;
        }

        _timesteps = new float[numInferenceSteps];
        float skippingStep = (float)_originalInferenceSteps / numInferenceSteps;
        for (int i = 0; i < numInferenceSteps; i++)
        {
            int idx = Math.Min((int)(skippingStep * (i + 1)) - 1, _originalInferenceSteps - 1);
            _timesteps[numInferenceSteps - 1 - i] = originTimesteps[idx];
        }
    }

    /// <summary>Performs one TCD step. Recovers x0 from the epsilon prediction, applies the consistency boundary scaling, jumps to the intermediate state s, then for non-final steps with Eta &gt; 0 adds stochasticity toward the next timestep.</summary>
    public void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex)
    {
        int timestep = (int)_timesteps[stepIndex];
        bool isLast = stepIndex + 1 >= _numInferenceSteps;
        int prevTimestep = isLast ? 0 : (int)_timesteps[stepIndex + 1];

        // Intermediate state s controlled by Eta: floor((1-eta)·t_prev).
        int timestepS = (int)MathF.Floor((1.0f - Eta) * prevTimestep);

        float alphaProdT = _alphasCumprod[timestep];
        float alphaProdTPrev = _alphasCumprod[prevTimestep];
        float alphaProdS = _alphasCumprod[timestepS];
        float betaProdT = 1.0f - alphaProdT;
        float betaProdS = 1.0f - alphaProdS;

        float sqrtAlphaProdT = MathF.Sqrt(alphaProdT);
        float sqrtBetaProdT = MathF.Sqrt(betaProdT);
        float sqrtAlphaProdS = MathF.Sqrt(alphaProdS);
        float sqrtBetaProdS = MathF.Sqrt(betaProdS);

        float scaledTimestep = timestep * TimestepScaling;
        float denom = scaledTimestep * scaledTimestep + SigmaData * SigmaData;
        float cSkip = (SigmaData * SigmaData) / denom;
        float cOut = scaledTimestep / MathF.Sqrt(denom);

        // Stochastic re-noise from s to t_prev (only when Eta > 0 and not the last step).
        bool stochastic = Eta > 0.0f && !isLast;
        float renoiseSignal = stochastic ? MathF.Sqrt(alphaProdTPrev / alphaProdS) : 0.0f;
        float renoiseNoise = stochastic ? MathF.Sqrt(1.0f - alphaProdTPrev / alphaProdS) : 0.0f;

        int count = (int)sample.ElementCount;
        Tensor? noise = stochastic ? SeedGenerator.CreateNoise(sample.Shape, unchecked(Seed * 9973 + stepIndex + 1)) : null;
        try
        {
            float* modelPtr = (float*)modelOutput.DataPointer;
            float* samplePtr = (float*)sample.DataPointer;
            float* outPtr = (float*)output.DataPointer;
            float* noisePtr = noise is null ? null : (float*)noise.DataPointer;

            for (int i = 0; i < count; i++)
            {
                float eps = modelPtr[i];
                float predX0 = (samplePtr[i] - sqrtBetaProdT * eps) / sqrtAlphaProdT;
                float denoised = cOut * predX0 + cSkip * samplePtr[i];
                float predNoisedS = sqrtAlphaProdS * denoised + sqrtBetaProdS * eps;
                outPtr[i] = noisePtr is null
                    ? predNoisedS
                    : renoiseSignal * predNoisedS + renoiseNoise * noisePtr[i];
            }
        }
        finally
        {
            noise?.Dispose();
        }
    }

    /// <summary>Adds noise to a clean sample (forward diffusion) for img2img starts.</summary>
    public void AddNoise(Tensor output, Tensor sample, Tensor noise, int stepIndex)
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
            float t = (float)i / (_numTrainTimesteps - 1);
            float sqrtBeta = MathF.Sqrt(_betaStart) + t * (MathF.Sqrt(_betaEnd) - MathF.Sqrt(_betaStart));
            float beta = sqrtBeta * sqrtBeta;
            cumprod *= 1.0f - beta;
            _alphasCumprod[i] = cumprod;
        }
    }
}
