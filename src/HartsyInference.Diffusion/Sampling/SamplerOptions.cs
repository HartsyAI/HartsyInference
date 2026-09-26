using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Family facts a sampler cannot derive from its sigma array, plus test seams.</summary>
public sealed record SamplerOptions
{
    /// <summary>Default options: no family mapping, seeded noise.</summary>
    public static SamplerOptions Default { get; } = new();

    /// <summary>ComfyUI's <c>model_sampling.percent_to_sigma</c> for the family; null derives it from the sigma array.</summary>
    public Func<double, double>? PercentToSigma { get; init; }

    /// <summary>Replaces the sampler's own noise source; parity tests inject a reference run's exact draws.</summary>
    public Func<TensorShape, float[], INoiseSource>? NoiseFactory { get; init; }

    /// <summary>Maps a percent of the schedule to sigma using <see cref="PercentToSigma"/> when set, otherwise a
    /// shifted-flow or log-linear fit to <paramref name="sigmas"/>.</summary>
    public double SigmaAtPercent(double percent, float[] sigmas, bool flow)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (PercentToSigma is not null)
        {
            return PercentToSigma(percent);
        }
        return flow ? FlowPercentToSigma(percent, EstimateFlowShift(sigmas)) : LogLinearPercentToSigma(percent, sigmas);
    }

    /// <summary>ComfyUI's flow <c>percent_to_sigma</c>: <c>time_snr_shift(shift, 1 − percent)</c>.</summary>
    public static double FlowPercentToSigma(double percent, double shift)
    {
        if (percent <= 0.0)
        {
            return 1.0;
        }
        if (percent >= 1.0)
        {
            return 0.0;
        }
        double t = 1.0 - percent;
        return shift * t / (1.0 + ((shift - 1.0) * t));
    }

    /// <summary>Recovers the shift of a <c>shift·t/(1+(shift−1)·t)</c> schedule from its middle entry; 1 if unusable.</summary>
    public static double EstimateFlowShift(float[] sigmas)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        int steps = sigmas.Length - 1;
        if (steps < 2)
        {
            return 1.0;
        }
        int mid = steps / 2;
        double t = 1.0 - ((double)mid / steps);
        double sigma = sigmas[mid];
        if (sigma <= 0.0 || sigma >= 1.0 || t <= 0.0 || t >= 1.0)
        {
            return 1.0;
        }
        double shift = sigma * (1.0 - t) / (t * (1.0 - sigma));
        return double.IsFinite(shift) && shift > 0.0 ? shift : 1.0;
    }

    private static double LogLinearPercentToSigma(double percent, float[] sigmas)
    {
        if (percent <= 0.0)
        {
            return sigmas[0];
        }
        if (percent >= 1.0)
        {
            return 0.0;
        }
        double max = sigmas[0];
        double min = max;
        for (int i = sigmas.Length - 1; i >= 0; i--)
        {
            if (sigmas[i] > 0f)
            {
                min = sigmas[i];
                break;
            }
        }
        return Math.Exp(Math.Log(max) + (percent * (Math.Log(min) - Math.Log(max))));
    }
}
