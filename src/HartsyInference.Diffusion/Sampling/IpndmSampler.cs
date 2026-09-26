using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>ipndm</c> (fixed Adams–Bashforth weights) and <c>ipndm_v</c> (variable-step weights): a
/// fourth-order explicit multistep method over the Karras derivative.</summary>
public sealed class IpndmSampler : SamplerBase
{
    private const int MaxOrder = 4;
    private readonly bool _variableStep;
    private readonly Tensor?[] _history = new Tensor?[MaxOrder - 1];
    private int _count;

    /// <summary>Creates the sampler; <paramref name="variableStep"/> selects <c>ipndm_v</c>.</summary>
    public IpndmSampler(float[] sigmas, bool variableStep, SamplerOptions? options = null)
        : base(sigmas, 0, options) => _variableStep = variableStep;

    /// <inheritdoc/>
    public override string Name => _variableStep ? "ipndm_v" : "ipndm";

    /// <inheritdoc/>
    protected override void OnReset()
    {
        for (int k = 0; k < _history.Length; k++)
        {
            Release(ref _history[k]);
        }
        _count = 0;
    }

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
            int order = Math.Min(MaxOrder, i + 1);
            if (sigmaNext == 0f)
            {
                backend.Scale(z, denoised, 1.0f);
            }
            else
            {
                Span<double> c = stackalloc double[MaxOrder];
                Coefficients(order, i, c);
                double dt = sigmaNext - sigma;
                SamplerOps.MixInto(backend, z, d, 1.0f, (float)(dt * c[0]));
                for (int k = 1; k < order; k++)
                {
                    SamplerOps.MixInto(backend, z, _history[k - 1]!, 1.0f, (float)(dt * c[k]));
                }
            }
            Push(backend, d);
            d = null!;
        }
        finally
        {
            d?.Dispose();
        }
    }

    private void Push(IBackend backend, Tensor d)
    {
        Release(ref _history[^1]);
        for (int k = _history.Length - 1; k > 0; k--)
        {
            _history[k] = _history[k - 1];
        }
        _history[0] = null;
        Keep(backend, ref _history[0], d);
        _count = Math.Min(_count + 1, _history.Length);
    }

    /// <summary>Weights on <c>d_cur, d_{-1}, d_{-2}, d_{-3}</c>, before multiplication by the step size.</summary>
    private void Coefficients(int order, int i, Span<double> c)
    {
        c.Clear();
        if (order == 1)
        {
            c[0] = 1.0;
            return;
        }
        if (!_variableStep)
        {
            switch (order)
            {
                case 2:
                    c[0] = 1.5;
                    c[1] = -0.5;
                    break;
                case 3:
                    c[0] = 23.0 / 12.0;
                    c[1] = -16.0 / 12.0;
                    c[2] = 5.0 / 12.0;
                    break;
                default:
                    c[0] = 55.0 / 24.0;
                    c[1] = -59.0 / 24.0;
                    c[2] = 37.0 / 24.0;
                    c[3] = -9.0 / 24.0;
                    break;
            }
            return;
        }
        double hN = (double)Sigma(i + 1) - Sigma(i);
        double hN1 = (double)Sigma(i) - Sigma(i - 1);
        if (order == 2)
        {
            c[0] = (2.0 + (hN / hN1)) / 2.0;
            c[1] = -(hN / hN1) / 2.0;
            return;
        }
        double hN2 = (double)Sigma(i - 1) - Sigma(i - 2);
        double temp1 = (1.0 - (hN / (3.0 * (hN + hN1)) * (hN * (hN + hN1)) / (hN1 * (hN1 + hN2)))) / 2.0;
        if (order == 3)
        {
            c[0] = ((2.0 + (hN / hN1)) / 2.0) + temp1;
            c[1] = (-(hN / hN1) / 2.0) - ((1.0 + (hN1 / hN2)) * temp1);
            c[2] = temp1 * hN1 / hN2;
            return;
        }
        double hN3 = (double)Sigma(i - 2) - Sigma(i - 3);
        double temp2 = (((1.0 - (hN / (3.0 * (hN + hN1)))) / 2.0) + ((1.0 - (hN / (2.0 * (hN + hN1)))) * hN / (6.0 * (hN + hN1 + hN2))))
            * (hN * (hN + hN1) * (hN + hN1 + hN2)) / (hN1 * (hN1 + hN2) * (hN1 + hN2 + hN3));
        double ratio = hN1 * (hN1 + hN2) / (hN2 * (hN2 + hN3));
        c[0] = ((2.0 + (hN / hN1)) / 2.0) + temp1 + temp2;
        c[1] = (-(hN / hN1) / 2.0) - ((1.0 + (hN1 / hN2)) * temp1) - ((1.0 + (hN1 / hN2) + ratio) * temp2);
        c[2] = (temp1 * hN1 / hN2) + (((hN1 / hN2) + (ratio * (1.0 + (hN2 / hN3)))) * temp2);
        c[3] = -temp2 * ratio * hN1 / hN2;
    }
}
