using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>gradient_estimation</c>: Euler plus a <c>(γ−1)</c>-weighted correction from the change in
/// the Karras derivative between steps.</summary>
public sealed class GradientEstimationSampler : SamplerBase
{
    private const double Gamma = 2.0;
    private Tensor? _oldD;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public GradientEstimationSampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "gradient_estimation";

    /// <inheritdoc/>
    protected override void OnReset() => Release(ref _oldD);

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        Tensor d = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d, z, denoised, 1.0f / sigma, -1.0f / sigma);
        try
        {
            float dt = sigmaNext - sigma;
            if (sigmaNext == 0f)
            {
                backend.Scale(z, denoised, 1.0f);
            }
            else if (i >= 1 && _oldD is not null)
            {
                float correction = (float)((Gamma - 1.0) * dt);
                SamplerOps.MixInto(backend, z, d, _oldD, 1.0f, dt + correction, -correction);
            }
            else
            {
                SamplerOps.MixInto(backend, z, d, 1.0f, dt);
            }
            Keep(backend, ref _oldD, d);
            d = null!;
        }
        finally
        {
            d?.Dispose();
        }
    }
}
