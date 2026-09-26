using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>ddpm</c>: the ancestral DDPM posterior step, with the cumulative alphas taken from sigma as
/// <c>1/(σ²+1)</c>.</summary>
public sealed class DdpmSampler : SamplerBase
{
    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public DdpmSampler(float[] sigmas, int seed, SamplerOptions? options = null)
        : base(sigmas, seed, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "ddpm";

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Gaussian;

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        double sigma = Sigma(i);
        double sigmaNext = Sigma(i + 1);
        using Tensor denoised = Denoise(backend, predictor, z, (float)sigma, stepIndex);
        double alphaCumprod = 1.0 / ((sigma * sigma) + 1.0);
        double alphaCumprodPrev = 1.0 / ((sigmaNext * sigmaNext) + 1.0);
        double alpha = alphaCumprod / alphaCumprodPrev;
        // mu = sqrt(1/alpha)·(x_vp − (1−alpha)·eps/sqrt(1−ac)), with x_vp = x/sqrt(1+σ²) and eps = (x − D)/σ.
        double c1 = Math.Sqrt(1.0 / alpha);
        double c2 = c1 * (1.0 - alpha) / Math.Sqrt(1.0 - alphaCumprod);
        double xScale = (c1 / Math.Sqrt(1.0 + (sigma * sigma))) - (c2 / sigma);
        double dScale = c2 / sigma;
        double outScale = sigmaNext != 0.0 ? Math.Sqrt(1.0 + (sigmaNext * sigmaNext)) : 1.0;
        SamplerOps.MixInto(backend, z, denoised, (float)(xScale * outScale), (float)(dScale * outScale));
        if (sigmaNext > 0.0)
        {
            double noise = Math.Sqrt((1.0 - alpha) * (1.0 - alphaCumprodPrev) / (1.0 - alphaCumprod));
            AddNoise(backend, z, stepIndex, 0, (float)sigma, (float)sigmaNext, (float)(noise * outScale));
        }
    }
}
