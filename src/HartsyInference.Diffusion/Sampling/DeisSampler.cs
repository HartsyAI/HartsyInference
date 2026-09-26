using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>deis</c> ('tab' mode, order 3): exponential-integrator multistep weights integrated
/// numerically in VP time once per run.</summary>
public sealed class DeisSampler : SamplerBase
{
    private const int MaxOrder = 3;
    private const int IntegrationPoints = 10000;
    private readonly Tensor?[] _history = new Tensor?[MaxOrder - 1];
    private double[][] _coefficients = [];

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public DeisSampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "deis";

    /// <inheritdoc/>
    protected override void OnReset()
    {
        for (int k = 0; k < _history.Length; k++)
        {
            Release(ref _history[k]);
        }
        _coefficients = [];
    }

    /// <inheritdoc/>
    protected override void OnBegin() => _coefficients = BuildCoefficients(LocalSigmas);

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
            int order = sigmaNext <= 0f ? 1 : Math.Min(MaxOrder, i + 1);
            if (order == 1)
            {
                SamplerOps.MixInto(backend, z, d, 1.0f, sigmaNext - sigma);
            }
            else
            {
                double[] c = _coefficients[i];
                SamplerOps.MixInto(backend, z, d, 1.0f, (float)c[0]);
                for (int k = 1; k < order; k++)
                {
                    SamplerOps.MixInto(backend, z, _history[k - 1]!, 1.0f, (float)c[k]);
                }
            }
            Release(ref _history[^1]);
            for (int k = _history.Length - 1; k > 0; k--)
            {
                _history[k] = _history[k - 1];
            }
            _history[0] = null;
            Keep(backend, ref _history[0], d);
            d = null!;
        }
        finally
        {
            d?.Dispose();
        }
    }

    /// <summary>Per-step weights on <c>d_cur, d_{-1}, d_{-2}</c>, from ComfyUI's <c>get_deis_coeff_list</c>.</summary>
    internal static double[][] BuildCoefficients(float[] sigmas)
    {
        // edm2t with ComfyUI's fixed epsilon_s = 1e-3, sigma_min = 0.002, sigma_max = 80.
        const double EpsilonS = 1e-3;
        double logMin = Math.Log((0.002 * 0.002) + 1.0);
        double logMax = Math.Log((80.0 * 80.0) + 1.0);
        double betaD = 2.0 * ((logMin / EpsilonS) - logMax) / (EpsilonS - 1.0);
        double beta0 = logMax - (0.5 * betaD);
        double beta1 = betaD + beta0;
        double[] t = new double[sigmas.Length];
        for (int k = 0; k < t.Length; k++)
        {
            double s = sigmas[k];
            t[k] = (Math.Sqrt((beta0 * beta0) + (2.0 * betaD * Math.Log((s * s) + 1.0))) - beta0) / betaD;
        }

        double[][] result = new double[sigmas.Length - 1][];
        for (int i = 0; i < result.Length; i++)
        {
            int order = Math.Min(i + 1, MaxOrder);
            if (order == 1 || t[i + 1] <= 0.0)
            {
                result[i] = [];
                continue;
            }
            double tCur = t[i];
            double tNext = t[i + 1];
            double dTau = (tNext - tCur) / IntegrationPoints;
            double[] coeff = new double[order];
            for (int n = 0; n < IntegrationPoints; n++)
            {
                double tau = tCur + ((tNext - tCur) * n / (IntegrationPoints - 1));
                double logAlpha = (-0.5 * tau * tau * (beta1 - beta0)) - (tau * beta0);
                double alpha = Math.Exp(logAlpha);
                double dLogAlpha = (-tau * (beta1 - beta0)) - beta0;
                double integrand = -0.5 * dLogAlpha / Math.Sqrt(alpha * (1.0 - alpha));
                for (int j = 0; j < order; j++)
                {
                    double poly = 1.0;
                    for (int k = 0; k < order; k++)
                    {
                        if (k != j)
                        {
                            poly *= (tau - t[i - k]) / (t[i - j] - t[i - k]);
                        }
                    }
                    coeff[j] += integrand * poly * dTau;
                }
            }
            result[i] = coeff;
        }
        return result;
    }
}
