using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>dpmpp_2m_sde</c> (midpoint solver) in half-log-SNR time, with Brownian noise so draws
/// over overlapping intervals correlate as ComfyUI's do.</summary>
public sealed class DpmPlusPlus2MSdeSampler : SamplerBase
{
    private readonly float _eta;
    private Tensor? _previousDenoised;
    private double _previousH;

    /// <summary>Creates the sampler; <paramref name="eta"/> 1.0 is ComfyUI's default.</summary>
    public DpmPlusPlus2MSdeSampler(float[] sigmas, int seed, float eta = 1.0f, SamplerOptions? options = null)
        : base(sigmas, seed, options) => _eta = eta;

    /// <inheritdoc/>
    public override string Name => "dpmpp_2m_sde";

    /// <inheritdoc/>
    protected override bool OffsetsFirstSigma => true;

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Brownian;

    /// <inheritdoc/>
    protected override void OnReset()
    {
        Release(ref _previousDenoised);
        _previousH = 0.0;
    }

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        try
        {
            if (sigmaNext == 0f)
            {
                backend.Scale(z, denoised, 1.0f);
            }
            else
            {
                double h = Lambda(sigmaNext) - Lambda(sigma);
                double hEta = h * (_eta + 1.0);
                double alphaT = sigmaNext * Math.Exp(Lambda(sigmaNext));
                double blend = -SamplerMath.Expm1(-hEta);
                float xScale = (float)(sigmaNext / sigma * Math.Exp(-h * _eta));
                if (_previousDenoised is null)
                {
                    SamplerOps.MixInto(backend, z, denoised, xScale, (float)(alphaT * blend));
                }
                else
                {
                    double c = 0.5 * alphaT * blend / (_previousH / h);
                    SamplerOps.MixInto(backend, z, denoised, _previousDenoised, xScale, (float)((alphaT * blend) + c), (float)-c);
                }
                if (_eta > 0f)
                {
                    double noise = sigmaNext * Math.Sqrt(-SamplerMath.Expm1(-2.0 * h * _eta));
                    AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, (float)noise);
                }
                _previousH = h;
            }
            Keep(backend, ref _previousDenoised, denoised);
            denoised = null!;
        }
        finally
        {
            denoised?.Dispose();
        }
    }
}
