using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>dpmpp_sde</c>: stochastic DPM-Solver++ with a midpoint evaluation (two model calls per
/// step) and two overlapping Brownian draws.</summary>
public sealed class DpmPlusPlusSdeSampler : SamplerBase
{
    private const double R = 0.5;
    private readonly float _eta;

    /// <summary>Creates the sampler; <paramref name="eta"/> 1.0 is ComfyUI's default.</summary>
    public DpmPlusPlusSdeSampler(float[] sigmas, int seed, float eta = 1.0f, SamplerOptions? options = null)
        : base(sigmas, seed, options) => _eta = eta;

    /// <inheritdoc/>
    public override string Name => "dpmpp_sde";

    /// <inheritdoc/>
    protected override bool OffsetsFirstSigma => true;

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Brownian;

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        if (sigmaNext == 0f)
        {
            backend.Scale(z, denoised, 1.0f);
            return;
        }
        double lambdaS = Lambda(sigma);
        double lambdaT = Lambda(sigmaNext);
        double h = lambdaT - lambdaS;
        double lambdaS1 = lambdaS + (R * h);
        double fac = 1.0 / (2.0 * R);
        double sigmaS1 = SigmaOfLambda(lambdaS1);
        double alphaS = sigma * Math.Exp(lambdaS);
        double alphaS1 = sigmaS1 * Math.Exp(lambdaS1);
        double alphaT = sigmaNext * Math.Exp(lambdaT);

        (double sd1, double su1) = SamplerMath.AncestralStep(Math.Exp(-lambdaS), Math.Exp(-lambdaS1), _eta);
        double h1 = -Math.Log(sd1) - lambdaS;
        using Tensor x2 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, x2, z, denoised, (float)(alphaS1 / alphaS * Math.Exp(-h1)), (float)(-alphaS1 * SamplerMath.Expm1(-h1)));
        if (_eta > 0f)
        {
            AddNoise(backend, x2, stepIndex, 0, sigma, (float)sigmaS1, (float)(alphaS1 * su1));
        }
        using Tensor denoised2 = Denoise(backend, predictor, x2, (float)sigmaS1, stepIndex);

        (double sd2, double su2) = SamplerMath.AncestralStep(Math.Exp(-lambdaS), Math.Exp(-lambdaT), _eta);
        double h2 = -Math.Log(sd2) - lambdaS;
        double blend = -alphaT * SamplerMath.Expm1(-h2);
        SamplerOps.MixInto(backend, z, denoised, denoised2, (float)(alphaT / alphaS * Math.Exp(-h2)), (float)(blend * (1.0 - fac)), (float)(blend * fac));
        if (_eta > 0f)
        {
            AddNoise(backend, z, stepIndex, 1, sigma, sigmaNext, (float)(alphaT * su2));
        }
    }
}
