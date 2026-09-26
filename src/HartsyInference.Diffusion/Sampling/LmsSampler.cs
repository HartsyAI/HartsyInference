using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>lms</c>: a linear multistep method weighting up to four past derivatives by integrated
/// Lagrange bases over sigma.</summary>
public sealed class LmsSampler : SamplerBase
{
    private const int Order = 4;
    private readonly Tensor?[] _derivatives = new Tensor?[Order];
    private int _count;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public LmsSampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "lms";

    /// <inheritdoc/>
    protected override void OnReset()
    {
        for (int k = 0; k < Order; k++)
        {
            Release(ref _derivatives[k]);
        }
        _count = 0;
    }

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        using Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        Tensor derivative = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, derivative, z, denoised, 1.0f / sigma, -1.0f / sigma);
        // Newest first; the oldest falls off once the window is full.
        Release(ref _derivatives[Order - 1]);
        for (int k = Order - 1; k > 0; k--)
        {
            _derivatives[k] = _derivatives[k - 1];
        }
        _derivatives[0] = null;
        Keep(backend, ref _derivatives[0], derivative);
        _count = Math.Min(_count + 1, Order);

        if (Sigma(i + 1) == 0f)
        {
            backend.Scale(z, denoised, 1.0f);
            return;
        }
        int order = Math.Min(i + 1, Order);
        for (int j = 0; j < order; j++)
        {
            SamplerOps.MixInto(backend, z, _derivatives[j]!, 1.0f, LinearMultistepCoefficient(order, i, j));
        }
    }

    /// <summary>Integrates the <paramref name="j"/>-th Lagrange basis over the step with Simpson's rule.</summary>
    private float LinearMultistepCoefficient(int order, int i, int j)
    {
        const int Subintervals = 200;
        double lo = Sigma(i);
        double width = Sigma(i + 1) - lo;
        double step = width / Subintervals;
        double total = 0.0;
        for (int k = 0; k <= Subintervals; k++)
        {
            double tau = lo + (k * step);
            double weight = k == 0 || k == Subintervals ? 1.0 : (k % 2 == 1 ? 4.0 : 2.0);
            total += weight * Basis(tau, order, i, j);
        }
        return (float)(total * step / 3.0);
    }

    private double Basis(double tau, int order, int i, int j)
    {
        double product = 1.0;
        for (int k = 0; k < order; k++)
        {
            if (k == j)
            {
                continue;
            }
            product *= (tau - Sigma(i - k)) / (Sigma(i - j) - Sigma(i - k));
        }
        return product;
    }
}
