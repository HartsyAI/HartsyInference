using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>dpm_2</c>, and with <c>eta &gt; 0</c> <c>dpm_2_ancestral</c>: step to the geometric
/// midpoint in sigma, re-evaluate there, take the whole step with that derivative.</summary>
public sealed class Dpm2Sampler : SamplerBase
{
    private readonly float _eta;

    /// <summary>Creates the sampler; <paramref name="eta"/> above zero selects the ancestral variant.</summary>
    public Dpm2Sampler(float[] sigmas, int seed, float eta = 0f, SamplerOptions? options = null)
        : base(sigmas, seed, options) => _eta = eta;

    /// <inheritdoc/>
    public override string Name => _eta > 0f ? "dpm_2_ancestral" : "dpm_2";

    /// <inheritdoc/>
    protected override NoiseKind Noise => _eta > 0f ? NoiseKind.Gaussian : NoiseKind.None;

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        bool flowAncestral = _eta > 0f && IsFlow;
        double rescale = 1.0;
        double renoise = 0.0;
        float sigmaDown;
        float sigmaUp = 0f;
        if (flowAncestral)
        {
            (double down, double r, double n) = SamplerMath.FlowAncestralStep(sigma, sigmaNext, _eta);
            sigmaDown = (float)down;
            rescale = r;
            renoise = n;
        }
        else
        {
            (sigmaDown, sigmaUp) = _eta > 0f ? SamplerMath.AncestralStep(sigma, sigmaNext, _eta) : (sigmaNext, 0f);
        }

        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        using Tensor d = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d, z, denoised, 1.0f / sigma, -1.0f / sigma);
        if (sigmaDown <= 0f)
        {
            SamplerOps.MixInto(backend, z, d, 1.0f, sigmaDown - sigma);
            return;
        }

        float sigmaMid = MathF.Exp(0.5f * (SamplerOps.LogSigma(sigma) + SamplerOps.LogSigma(sigmaDown)));
        using Tensor zMid = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, zMid, z, d, 1.0f, sigmaMid - sigma);
        using Tensor denoisedMid = Denoise(backend, predictor, zMid, sigmaMid, stepIndex);
        using Tensor dMid = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, dMid, zMid, denoisedMid, 1.0f / sigmaMid, -1.0f / sigmaMid);
        SamplerOps.MixInto(backend, z, dMid, 1.0f, sigmaDown - sigma);
        if (flowAncestral)
        {
            SamplerOps.ScaleInPlace(backend, z, (float)rescale);
            AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, (float)renoise);
        }
        else if (sigmaUp > 0f)
        {
            AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, sigmaUp);
        }
    }
}
