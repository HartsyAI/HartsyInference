using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>res_multistep</c> (eta 0): the second-order RES multistep exponential integrator.</summary>
public sealed class ResMultistepSampler : SamplerBase
{
    private Tensor? _oldDenoised;
    private double _oldSigmaDown;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public ResMultistepSampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "res_multistep";

    /// <inheritdoc/>
    protected override void OnReset() => Release(ref _oldDenoised);

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        double sigmaDown = Sigma(i + 1);
        Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        try
        {
            if (sigmaDown == 0.0 || _oldDenoised is null)
            {
                float dt = (float)((sigmaDown - sigma) / sigma);
                SamplerOps.MixInto(backend, z, denoised, 1.0f + dt, -dt);
            }
            else
            {
                double t = -Math.Log(sigma);
                double tOld = -Math.Log(_oldSigmaDown);
                double tNext = -Math.Log(sigmaDown);
                double tPrev = -Math.Log(Sigma(i - 1));
                double h = tNext - t;
                double c2 = (tPrev - tOld) / h;
                double phi1 = SamplerMath.Expm1(-h) / -h;
                double phi2 = (phi1 - 1.0) / -h;
                double b1 = FiniteOrZero(phi1 - (phi2 / c2));
                double b2 = FiniteOrZero(phi2 / c2);
                SamplerOps.MixInto(backend, z, denoised, _oldDenoised, (float)Math.Exp(-h), (float)(h * b1), (float)(h * b2));
            }
            _oldSigmaDown = sigmaDown;
            Keep(backend, ref _oldDenoised, denoised);
            denoised = null!;
        }
        finally
        {
            denoised?.Dispose();
        }
    }

    private static double FiniteOrZero(double value) => double.IsFinite(value) ? value : 0.0;
}
