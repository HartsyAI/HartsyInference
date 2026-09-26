using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>dpmpp_2s_ancestral</c>: a DPM-Solver++(2S) midpoint step then fresh noise, with ComfyUI's
/// rectified-flow form on flow models.</summary>
public sealed class DpmPlusPlus2SAncestralSampler : SamplerBase
{
    private readonly float _eta;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public DpmPlusPlus2SAncestralSampler(float[] sigmas, int seed, float eta = 1.0f, SamplerOptions? options = null)
        : base(sigmas, seed, options) => _eta = eta;

    /// <inheritdoc/>
    public override string Name => "dpmpp_2s_ancestral";

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Gaussian;

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        if (IsFlow)
        {
            StepFlow(backend, z, predictor, i, stepIndex);
            return;
        }
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        (float sigmaDown, float sigmaUp) = SamplerMath.AncestralStep(sigma, sigmaNext, _eta);
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        if (sigmaDown <= 0f)
        {
            float dtTerminal = (sigmaDown - sigma) / sigma;
            SamplerOps.MixInto(backend, z, denoised, 1.0f + dtTerminal, -dtTerminal);
        }
        else
        {
            float lambda = SamplerOps.Lambda(sigma);
            float h = SamplerOps.Lambda(sigmaDown) - lambda;
            float sigmaMid = MathF.Exp(-(lambda + (0.5f * h)));
            using Tensor zMid = new Tensor(z.Shape, DType.F32);
            SamplerOps.SetMix(backend, zMid, z, denoised, sigmaMid / sigma, 1.0f - MathF.Exp(-0.5f * h));
            using Tensor denoisedMid = Denoise(backend, predictor, zMid, sigmaMid, stepIndex);
            SamplerOps.MixInto(backend, z, denoisedMid, sigmaDown / sigma, 1.0f - MathF.Exp(-h));
        }
        if (sigmaNext > 0f && sigmaUp > 0f)
        {
            AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, sigmaUp);
        }
    }

    private void StepFlow(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        (double sigmaDown, double rescale, double renoise) = SamplerMath.FlowAncestralStep(sigma, sigmaNext, _eta);
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        if (sigmaNext == 0f)
        {
            float dt = (float)((sigmaDown - sigma) / sigma);
            SamplerOps.MixInto(backend, z, denoised, 1.0f + dt, -dt);
            return;
        }
        double sigmaS;
        if (sigma == 1.0f)
        {
            sigmaS = 0.9999;
        }
        else
        {
            double tI = Math.Log((1.0 - sigma) / sigma);
            double tDown = Math.Log((1.0 - sigmaDown) / sigmaDown);
            sigmaS = 1.0 / (Math.Exp(tI + (0.5 * (tDown - tI))) + 1.0);
        }
        double sRatio = sigmaS / sigma;
        using Tensor u = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, u, z, denoised, (float)sRatio, (float)(1.0 - sRatio));
        using Tensor denoisedS = Denoise(backend, predictor, u, (float)sigmaS, stepIndex);
        double dRatio = sigmaDown / sigma;
        SamplerOps.MixInto(backend, z, denoisedS, (float)dRatio, (float)(1.0 - dRatio));
        if (_eta > 0f)
        {
            SamplerOps.ScaleInPlace(backend, z, (float)rescale);
            AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, (float)renoise);
        }
    }
}
