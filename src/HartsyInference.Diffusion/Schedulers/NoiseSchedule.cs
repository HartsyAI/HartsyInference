using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Computes beta, alpha, and cumulative alpha schedules shared by all diffusion schedulers.</summary>
public static class NoiseSchedule
{
    /// <summary>Forward diffusion for img2img starts, shared by the discrete VP schedulers: <c>noisy = sqrt(ᾱ_t)·sample + sqrt(1-ᾱ_t)·noise</c> with <c>t = timesteps[stepIndex]</c>.</summary>
    public static unsafe void AddNoise(Tensor output, Tensor sample, Tensor noise,
        ReadOnlySpan<float> timesteps, float[] alphasCumprod, int stepIndex)
    {
        int timestep = (int)timesteps[stepIndex];
        float sqrtAlphaCumprod = MathF.Sqrt(alphasCumprod[timestep]);
        float sqrtOneMinusAlphaCumprod = MathF.Sqrt(1.0f - alphasCumprod[timestep]);

        float* samplePtr = (float*)sample.DataPointer;
        float* noisePtr = (float*)noise.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        for (int i = 0; i < count; i++)
        {
            outPtr[i] = sqrtAlphaCumprod * samplePtr[i] + sqrtOneMinusAlphaCumprod * noisePtr[i];
        }
    }

    /// <summary>Linear or, for <see cref="BetaScheduleType.ScaledLinear"/>, sqrt-interpolated-then-squared beta schedule, per <see cref="SchedulerConfig.BetaSchedule"/>.</summary>
    public static float[] ComputeBetas(SchedulerConfig config)
    {
        float[] betas = new float[config.NumTrainTimesteps];

        switch (config.BetaSchedule)
        {
            case BetaScheduleType.Linear:
                for (int i = 0; i < config.NumTrainTimesteps; i++)
                {
                    betas[i] = config.BetaStart + (config.BetaEnd - config.BetaStart) * i / (config.NumTrainTimesteps - 1);
                }
                break;

            case BetaScheduleType.ScaledLinear:
                float sqrtStart = MathF.Sqrt(config.BetaStart);
                float sqrtEnd = MathF.Sqrt(config.BetaEnd);
                for (int i = 0; i < config.NumTrainTimesteps; i++)
                {
                    float sqrtBeta = sqrtStart + (sqrtEnd - sqrtStart) * i / (config.NumTrainTimesteps - 1);
                    betas[i] = sqrtBeta * sqrtBeta;
                }
                break;

            default:
                throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                    $"Unsupported beta schedule type: {config.BetaSchedule}");
        }

        return betas;
    }

    /// <summary>Computes alphas = 1 - betas.</summary>
    public static float[] ComputeAlphas(float[] betas)
    {
        float[] alphas = new float[betas.Length];
        for (int i = 0; i < betas.Length; i++)
        {
            alphas[i] = 1.0f - betas[i];
        }
        return alphas;
    }

    /// <summary>Running product <c>ᾱ_i = Π(alphas[0..i])</c>.</summary>
    public static float[] ComputeAlphasCumprod(float[] alphas)
    {
        float[] alphasCumprod = new float[alphas.Length];
        alphasCumprod[0] = alphas[0];
        for (int i = 1; i < alphas.Length; i++)
        {
            alphasCumprod[i] = alphasCumprod[i - 1] * alphas[i];
        }
        return alphasCumprod;
    }

    /// <summary>Computes sigma values from alphas_cumprod: sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod).</summary>
    public static float[] ComputeSigmas(float[] alphasCumprod)
    {
        float[] sigmas = new float[alphasCumprod.Length];
        for (int i = 0; i < alphasCumprod.Length; i++)
        {
            sigmas[i] = MathF.Sqrt((1.0f - alphasCumprod[i]) / alphasCumprod[i]);
        }
        return sigmas;
    }

    /// <summary>Computes Karras sigma schedule (Karras et al. 2022).</summary>
    /// <param name="sigmaMin">Minimum sigma value.</param>
    /// <param name="sigmaMax">Maximum sigma value.</param>
    /// <param name="numSteps">Number of inference steps.</param>
    /// <param name="rho">Karras rho parameter. Default: 7.0.</param>
    public static float[] ComputeKarrasSigmas(float sigmaMin, float sigmaMax, int numSteps, float rho = 7.0f)
    {
        float[] sigmas = new float[numSteps + 1];
        float minInvRho = MathF.Pow(sigmaMin, 1.0f / rho);
        float maxInvRho = MathF.Pow(sigmaMax, 1.0f / rho);

        for (int i = 0; i < numSteps; i++)
        {
            float ramp = (float)i / (numSteps - 1);
            sigmas[i] = MathF.Pow(maxInvRho + ramp * (minInvRho - maxInvRho), rho);
        }
        sigmas[numSteps] = 0.0f;

        return sigmas;
    }

    /// <summary>Selects inference timesteps from the full training schedule.</summary>
    public static float[] SelectTimesteps(int numTrainTimesteps, int numInferenceSteps, TimestepSpacing spacing)
    {
        float[] timesteps;

        switch (spacing)
        {
            case TimestepSpacing.Linspace:
                timesteps = new float[numInferenceSteps];
                for (int i = 0; i < numInferenceSteps; i++)
                {
                    float t = (float)(numTrainTimesteps - 1) * (1.0f - (float)i / (numInferenceSteps - 1));
                    timesteps[i] = MathF.Round(t);
                }
                break;

            case TimestepSpacing.Leading:
                int stepRatio = numTrainTimesteps / numInferenceSteps;
                timesteps = new float[numInferenceSteps];
                for (int i = 0; i < numInferenceSteps; i++)
                {
                    timesteps[i] = MathF.Round((numInferenceSteps - 1 - i) * stepRatio);
                }
                break;

            case TimestepSpacing.Trailing:
                float trailingRatio = (float)numTrainTimesteps / numInferenceSteps;
                timesteps = new float[numInferenceSteps];
                for (int i = 0; i < numInferenceSteps; i++)
                {
                    timesteps[i] = MathF.Round(numTrainTimesteps - 1 - i * trailingRatio);
                }
                break;

            default:
                throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                    $"Unsupported timestep spacing: {spacing}");
        }

        return timesteps;
    }
}
