namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>A discrete-time sampler over latents held as flat host arrays: one denoising step per call, in schedule order.</summary>
public interface IHunyuanPaintSampler
{
    /// <summary>Integer timestep the model is called with at each step.</summary>
    IReadOnlyList<int> Timesteps { get; }

    /// <summary>Standard deviation the initial noise is scaled by.</summary>
    double InitialNoiseScale { get; }

    /// <summary>The model input for step <paramref name="step"/> (the sample itself, or divided by <c>sqrt(sigma^2 + 1)</c>).</summary>
    float[] ScaleModelInput(float[] sample, int step);

    /// <summary>Advances one step with the model's v-prediction; <paramref name="noise"/> feeds ancestral samplers and may be null for deterministic ones.</summary>
    float[] Step(float[] modelOutput, float[] sample, int step, float[]? noise);
}
