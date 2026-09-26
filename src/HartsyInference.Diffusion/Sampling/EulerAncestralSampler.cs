using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>euler_ancestral</c>: an Euler move to <c>sigmaDown</c> followed by fresh noise, with
/// ComfyUI's rectified-flow form on flow models.</summary>
public sealed class EulerAncestralSampler : SamplerBase
{
    private readonly float _eta;

    /// <summary>Creates the sampler; <paramref name="eta"/> 1.0 is ComfyUI's default.</summary>
    public EulerAncestralSampler(float[] sigmas, int seed, float eta = 1.0f, SamplerOptions? options = null)
        : base(sigmas, seed, options) => _eta = eta;

    /// <inheritdoc/>
    public override string Name => "euler_ancestral";

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Gaussian;

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        if (IsFlow)
        {
            if (sigmaNext == 0f)
            {
                backend.Scale(z, denoised, 1.0f);
                return;
            }
            (double down, double rescale, double renoise) = SamplerMath.FlowAncestralStep(sigma, sigmaNext, _eta);
            double ratio = down / sigma;
            SamplerOps.MixInto(backend, z, denoised, (float)ratio, (float)(1.0 - ratio));
            if (_eta > 0f)
            {
                SamplerOps.ScaleInPlace(backend, z, (float)rescale);
                AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, (float)renoise);
            }
            return;
        }
        (float sigmaDown, float sigmaUp) = SamplerMath.AncestralStep(sigma, sigmaNext, _eta);
        float dt = (sigmaDown - sigma) / sigma;
        SamplerOps.MixInto(backend, z, denoised, 1.0f + dt, -dt);
        if (sigmaUp > 0f)
        {
            AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, sigmaUp);
        }
    }
}
