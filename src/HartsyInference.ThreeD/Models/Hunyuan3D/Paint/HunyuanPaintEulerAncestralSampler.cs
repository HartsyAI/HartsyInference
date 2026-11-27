namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>diffusers 0.30 <c>EulerAncestralDiscreteScheduler</c> as Paint 2.0 configures it: v-prediction, trailing timesteps, the model input divided by
/// <c>sqrt(sigma^2 + 1)</c>, the initial noise scaled by the largest sigma.</summary>
public sealed class HunyuanPaintEulerAncestralSampler : IHunyuanPaintSampler
{
    private readonly int[] _timesteps;
    private readonly double[] _sigmas;

    /// <summary>Prepares <paramref name="steps"/> steps on <paramref name="schedule"/>.</summary>
    public HunyuanPaintEulerAncestralSampler(HunyuanPaintSchedule schedule, int steps)
    {
        ArgumentNullException.ThrowIfNull(schedule);
        _timesteps = HunyuanPaintSchedule.TrailingTimesteps(steps);
        _sigmas = schedule.SigmasAt(_timesteps, 0.0);
    }

    /// <inheritdoc/>
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <inheritdoc/>
    public double InitialNoiseScale => _sigmas.Max();

    /// <inheritdoc/>
    public float[] ScaleModelInput(float[] sample, int step)
    {
        double scale = 1.0 / Math.Sqrt(_sigmas[step] * _sigmas[step] + 1);
        float[] scaled = new float[sample.Length];
        for (int i = 0; i < scaled.Length; i++)
            scaled[i] = (float)(sample[i] * scale);
        return scaled;
    }

    /// <inheritdoc/>
    public float[] Step(float[] modelOutput, float[] sample, int step, float[]? noise)
    {
        if (noise is null || noise.Length != sample.Length)
            throw new ArgumentException("An ancestral step needs noise of the sample's size.", nameof(noise));
        double sigma = _sigmas[step], sigmaTo = _sigmas[step + 1];
        double sigmaUp = Math.Sqrt(sigmaTo * sigmaTo * (sigma * sigma - sigmaTo * sigmaTo) / (sigma * sigma));
        double sigmaDown = Math.Sqrt(sigmaTo * sigmaTo - sigmaUp * sigmaUp);
        double cOut = -sigma / Math.Sqrt(sigma * sigma + 1), cSkip = 1.0 / (sigma * sigma + 1), dt = sigmaDown - sigma;
        float[] result = new float[sample.Length];
        for (int i = 0; i < result.Length; i++)
        {
            double predicted = modelOutput[i] * cOut + sample[i] * cSkip;
            double derivative = (sample[i] - predicted) / sigma;
            result[i] = (float)(sample[i] + derivative * dt + noise[i] * sigmaUp);
        }
        return result;
    }
}
