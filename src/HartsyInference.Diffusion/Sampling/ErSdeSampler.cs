using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>er_sde</c>: the VP ER-SDE-Solver-3, a stage-3 extended reverse-time SDE solver.</summary>
public sealed class ErSdeSampler : SamplerBase
{
    private const int MaxStage = 3;
    private const int IntegrationPoints = 200;
    private double[] _erLambdas = [];
    private Tensor? _oldDenoised;
    private Tensor? _oldDenoisedD;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public ErSdeSampler(float[] sigmas, int seed, SamplerOptions? options = null)
        : base(sigmas, seed, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "er_sde";

    /// <inheritdoc/>
    protected override bool OffsetsFirstSigma => true;

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Gaussian;

    /// <inheritdoc/>
    protected override void OnReset()
    {
        Release(ref _oldDenoised);
        Release(ref _oldDenoisedD);
        _erLambdas = [];
    }

    /// <inheritdoc/>
    protected override void OnBegin()
    {
        _erLambdas = new double[LocalLength];
        for (int k = 0; k < LocalLength; k++)
        {
            double sigma = Sigma(k);
            _erLambdas[k] = sigma == 0.0 ? 0.0 : Math.Exp(-Lambda(sigma));
        }
    }

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        try
        {
            int stage = Math.Min(MaxStage, i + 1);
            if (sigmaNext == 0f)
            {
                backend.Scale(z, denoised, 1.0f);
            }
            else
            {
                double erS = _erLambdas[i];
                double erT = _erLambdas[i + 1];
                double alphaS = sigma / erS;
                double alphaT = sigmaNext / erT;
                double r = NoiseScaler(erT) / NoiseScaler(erS);
                SamplerOps.MixInto(backend, z, denoised, (float)(alphaT / alphaS * r), (float)(alphaT * (1.0 - r)));
                if (stage >= 2)
                {
                    double dt = erT - erS;
                    double step = -dt / IntegrationPoints;
                    double s = 0.0;
                    double su = 0.0;
                    for (int k = 0; k < IntegrationPoints; k++)
                    {
                        double pos = erT + (k * step);
                        double scaled = NoiseScaler(pos);
                        s += 1.0 / scaled;
                        su += (pos - erS) / scaled;
                    }
                    s *= step;
                    su *= step;
                    Tensor denoisedD = new Tensor(z.Shape, DType.F32);
                    double spanD = erS - _erLambdas[i - 1];
                    SamplerOps.SetMix(backend, denoisedD, denoised, _oldDenoised!, (float)(1.0 / spanD), (float)(-1.0 / spanD));
                    SamplerOps.MixInto(backend, z, denoisedD, 1.0f, (float)(alphaT * (dt + (s * NoiseScaler(erT)))));
                    if (stage >= 3 && _oldDenoisedD is not null)
                    {
                        double spanU = (erS - _erLambdas[i - 2]) / 2.0;
                        double coeffU = alphaT * ((dt * dt / 2.0) + (su * NoiseScaler(erT))) / spanU;
                        SamplerOps.MixInto(backend, z, denoisedD, _oldDenoisedD, 1.0f, (float)coeffU, (float)-coeffU);
                    }
                    Keep(backend, ref _oldDenoisedD, denoisedD);
                }
                double noiseSq = (erT * erT) - (erS * erS * r * r);
                double noise = noiseSq > 0.0 ? Math.Sqrt(noiseSq) : 0.0;
                AddNoise(backend, z, stepIndex, 0, sigma, sigmaNext, (float)(alphaT * noise));
            }
            Keep(backend, ref _oldDenoised, denoised);
            denoised = null!;
        }
        finally
        {
            denoised?.Dispose();
        }
    }

    private static double NoiseScaler(double x) => x * (Math.Exp(Math.Pow(x, 0.3)) + 10.0);
}
