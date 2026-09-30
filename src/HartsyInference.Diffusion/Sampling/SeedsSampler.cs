using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>seeds_2</c> and <c>seeds_3</c>: stochastic explicit exponential derivative-free solvers
/// with two or three model evaluations per step.</summary>
public sealed class SeedsSampler : SamplerBase
{
    private readonly int _stages;
    private readonly float _eta;

    /// <summary>Creates the sampler; <paramref name="stages"/> is 2 or 3.</summary>
    public SeedsSampler(float[] sigmas, int seed, int stages, float eta = 1.0f, SamplerOptions? options = null)
        : base(sigmas, seed, options)
    {
        if (stages is not (2 or 3))
        {
            throw new ArgumentOutOfRangeException(nameof(stages), "SEEDS has 2 or 3 stages.");
        }
        _stages = stages;
        _eta = eta;
    }

    /// <inheritdoc/>
    public override string Name => _stages == 2 ? "seeds_2" : "seeds_3";

    /// <inheritdoc/>
    protected override bool OffsetsFirstSigma => true;

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Gaussian;

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
        double hEta = h * (_eta + 1.0);
        double alphaT = sigmaNext * Math.Exp(lambdaT);
        bool injectNoise = _eta > 0f;
        double r1 = _stages == 2 ? 0.5 : 1.0 / 3.0;
        double lambdaS1 = lambdaS + (r1 * (lambdaT - lambdaS));
        double sigmaS1 = SigmaOfLambda(lambdaS1);
        double alphaS1 = sigmaS1 * Math.Exp(lambdaS1);

        using Tensor x2 = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, x2, z, denoised, (float)(sigmaS1 / sigma * Math.Exp(-r1 * h * _eta)),
            (float)(-alphaS1 * SamplerMath.PhiOne(-r1 * hEta)));
        Tensor? sdeNoise = null;
        try
        {
            if (injectNoise)
            {
                sdeNoise = DrawNoise(stepIndex, 0, sigma, (float)sigmaS1);
                SamplerOps.ScaleInPlace(backend, sdeNoise, (float)Math.Sqrt(-SamplerMath.Expm1(-2.0 * r1 * h * _eta)));
                SamplerOps.MixInto(backend, x2, sdeNoise, 1.0f, (float)sigmaS1);
            }
            using Tensor denoised2 = Denoise(backend, predictor, x2, (float)sigmaS1, stepIndex);
            float xScale = (float)(sigmaNext / sigma * Math.Exp(-h * _eta));
            double phiOneT = SamplerMath.PhiOne(-hEta);
            if (_stages == 2)
            {
                // phi_1 solver with r = 0.5: lerp(denoised, denoised2, 1/(2r)) is denoised2.
                double fac = 1.0 / (2.0 * r1);
                SamplerOps.MixInto(backend, z, denoised, denoised2, xScale, (float)(-alphaT * phiOneT * (1.0 - fac)), (float)(-alphaT * phiOneT * fac));
                if (injectNoise)
                {
                    AdvanceNoise(backend, ref sdeNoise, (r1 - 1.0) * h * _eta, stepIndex, 1, (float)sigmaS1, sigmaNext);
                    SamplerOps.MixInto(backend, z, sdeNoise!, 1.0f, sigmaNext);
                }
                return;
            }

            const double R2 = 2.0 / 3.0;
            double lambdaS2 = lambdaS + (R2 * (lambdaT - lambdaS));
            double sigmaS2 = SigmaOfLambda(lambdaS2);
            double alphaS2 = sigmaS2 * Math.Exp(lambdaS2);
            double a32 = R2 / r1 * SamplerMath.PhiTwo(-R2 * hEta);
            double a31 = SamplerMath.PhiOne(-R2 * hEta) - a32;
            using Tensor x3 = new Tensor(z.Shape, DType.F32);
            SamplerOps.SetMix(backend, x3, z, denoised, denoised2, (float)(sigmaS2 / sigma * Math.Exp(-R2 * h * _eta)),
                (float)(-alphaS2 * a31), (float)(-alphaS2 * a32));
            if (injectNoise)
            {
                AdvanceNoise(backend, ref sdeNoise, (r1 - R2) * h * _eta, stepIndex, 1, (float)sigmaS1, (float)sigmaS2);
                SamplerOps.MixInto(backend, x3, sdeNoise!, 1.0f, (float)sigmaS2);
            }
            using Tensor denoised3 = Denoise(backend, predictor, x3, (float)sigmaS2, stepIndex);
            double b3 = SamplerMath.PhiTwo(-hEta) / R2;
            double b1 = phiOneT - b3;
            SamplerOps.MixInto(backend, z, denoised, denoised3, xScale, (float)(-alphaT * b1), (float)(-alphaT * b3));
            if (injectNoise)
            {
                AdvanceNoise(backend, ref sdeNoise, (R2 - 1.0) * h * _eta, stepIndex, 2, (float)sigmaS2, sigmaNext);
                SamplerOps.MixInto(backend, z, sdeNoise!, 1.0f, sigmaNext);
            }
        }
        finally
        {
            sdeNoise?.Dispose();
        }
    }

    /// <summary><c>noise ← e^seg·noise + sqrt(−expm1(2·seg))·N</c>, carrying the accumulated SDE noise to the next stage.</summary>
    private void AdvanceNoise(IBackend backend, ref Tensor? sdeNoise, double segment, int stepIndex, int subDraw, float from, float to)
    {
        using Tensor fresh = DrawNoise(stepIndex, subDraw, from, to);
        SamplerOps.MixInto(backend, sdeNoise!, fresh, (float)Math.Exp(segment), (float)Math.Sqrt(-SamplerMath.Expm1(2.0 * segment)));
    }
}
