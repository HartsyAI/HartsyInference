using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>heunpp2</c>: Heun++ with a third evaluation two steps ahead, weighted by sigma relative to
/// the run's starting sigma; up to three model evaluations per step.</summary>
public sealed class HeunPlusPlus2Sampler : SamplerBase
{
    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public HeunPlusPlus2Sampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "heunpp2";

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        float sigmaEnd = Sigma(LocalLength - 1);
        float dt = sigmaNext - sigma;
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        using Tensor d = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d, z, denoised, 1.0f / sigma, -1.0f / sigma);
        if (sigmaNext == sigmaEnd)
        {
            SamplerOps.MixInto(backend, z, d, 1.0f, dt);
            return;
        }

        using Tensor x2 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, x2, z, d, 1.0f, dt);
        using Tensor denoised2 = Denoise(backend, predictor, x2, sigmaNext, stepIndex);
        using Tensor d2 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d2, x2, denoised2, 1.0f / sigmaNext, -1.0f / sigmaNext);
        double sigma0 = Sigma(0);
        float sigmaNext2 = Sigma(i + 2);
        if (sigmaNext2 == sigmaEnd)
        {
            double w2 = sigmaNext / (2.0 * sigma0);
            SamplerOps.MixInto(backend, z, d, d2, 1.0f, (float)((1.0 - w2) * dt), (float)(w2 * dt));
            return;
        }

        float dt2 = sigmaNext2 - sigmaNext;
        using Tensor x3 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, x3, x2, d2, 1.0f, dt2);
        using Tensor denoised3 = Denoise(backend, predictor, x3, sigmaNext2, stepIndex);
        using Tensor d3 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d3, x3, denoised3, 1.0f / sigmaNext2, -1.0f / sigmaNext2);
        double w = 3.0 * sigma0;
        double wB = sigmaNext / w;
        double wC = sigmaNext2 / w;
        SamplerOps.MixInto(backend, z, d, d2, 1.0f, (float)((1.0 - wB - wC) * dt), (float)(wB * dt));
        SamplerOps.MixInto(backend, z, d3, 1.0f, (float)(wC * dt));
    }
}
