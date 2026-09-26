using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>dpmpp_3m_sde</c>: third-order multistep DPM-Solver++ SDE in half-log-SNR time.</summary>
public sealed class DpmPlusPlus3MSdeSampler : SamplerBase
{
    private readonly float _eta;
    private Tensor? _denoised1;
    private Tensor? _denoised2;
    private double _h1;
    private double _h2;

    /// <summary>Creates the sampler; <paramref name="eta"/> 1.0 is ComfyUI's default.</summary>
    public DpmPlusPlus3MSdeSampler(float[] sigmas, int seed, float eta = 1.0f, SamplerOptions? options = null)
        : base(sigmas, seed, options) => _eta = eta;

    /// <inheritdoc/>
    public override string Name => "dpmpp_3m_sde";

    /// <inheritdoc/>
    protected override bool OffsetsFirstSigma => true;

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Brownian;

    /// <inheritdoc/>
    protected override void OnReset()
    {
        Release(ref _denoised1);
        Release(ref _denoised2);
        _h1 = 0.0;
        _h2 = 0.0;
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
                float xScale = (float)(sigmaNext / sigma * Math.Exp(-h * _eta));
                double cD = alphaT * -SamplerMath.Expm1(-hEta);
                double phi2 = (SamplerMath.Expm1(-hEta) / hEta) + 1.0;
                if (_denoised2 is not null)
                {
                    double r0 = _h1 / h;
                    double r1 = _h2 / h;
                    double k = r0 / (r0 + r1);
                    double m = 1.0 / (r0 + r1);
                    double phi3 = (phi2 / hEta) - 0.5;
                    double a = (phi2 * (1.0 + k)) - (phi3 * m);
                    double b = (-phi2 * k) + (phi3 * m);
                    SamplerOps.MixInto(backend, z, denoised, _denoised1!, xScale, (float)(cD + (alphaT * a / r0)),
                        (float)(alphaT * ((-a / r0) + (b / r1))));
                    SamplerOps.MixInto(backend, z, _denoised2, 1.0f, (float)(-alphaT * b / r1));
                }
                else if (_denoised1 is not null)
                {
                    double c = alphaT * phi2 / (_h1 / h);
                    SamplerOps.MixInto(backend, z, denoised, _denoised1, xScale, (float)(cD + c), (float)-c);
                }
                else
                {
                    SamplerOps.MixInto(backend, z, denoised, xScale, (float)cD);
                }
                if (_eta > 0f)
                {
                    double noise = sigmaNext * Math.Sqrt(-SamplerMath.Expm1(-2.0 * h * _eta));
                    AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, (float)noise);
                }
                _h2 = _h1;
                _h1 = h;
            }
            Release(ref _denoised2);
            _denoised2 = _denoised1;
            _denoised1 = null;
            Keep(backend, ref _denoised1, denoised);
            denoised = null!;
        }
        finally
        {
            denoised?.Dispose();
        }
    }
}
