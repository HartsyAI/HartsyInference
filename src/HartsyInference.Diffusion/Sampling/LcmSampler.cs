using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>lcm</c> (<c>sample_lcm</c>): jump to the denoised estimate, then re-noise it to the next sigma
/// with the model type's own noise scaling. Consistency-distilled checkpoints (LCM, Boogu Turbo) are trained for it.</summary>
public sealed class LcmSampler : SamplerBase
{
    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public LcmSampler(float[] sigmas, int seed, SamplerOptions? options = null)
        : base(sigmas, seed, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "lcm";

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Gaussian;

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        (float denoisedScale, float noiseScale) = Renoise(sigmaNext, IsFlow);
        backend.Scale(z, denoised, denoisedScale);
        if (noiseScale > 0f)
        {
            AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, noiseScale);
        }
    }

    /// <summary>ComfyUI's <c>noise_scaling(sigmaNext, noise, denoised)</c> as (denoised, noise) coefficients: CONST (flow)
    /// models interpolate, <c>(1 − σ')·D + σ'·ε</c>; EPS models add, <c>D + σ'·ε</c>. Zero noise at the terminal step.</summary>
    public static (float Denoised, float Noise) Renoise(float sigmaNext, bool flow)
    {
        if (sigmaNext <= 0f)
        {
            return (1f, 0f);
        }
        return flow ? (1f - sigmaNext, sigmaNext) : (1f, sigmaNext);
    }
}
