using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>lms</c> — linear multistep (Adams–Bashforth) over the sigma domain. One model evaluation
/// per step; the order comes from a weighted combination of the last <c>order</c> derivatives rather than from extra
/// forwards. This was the original Stable Diffusion default sampler and still appears throughout older workflows.
///
/// <para>The coefficients are not constants: Adams–Bashforth assumes evenly-spaced abscissae, and a diffusion sigma
/// schedule is anything but. They are therefore integrated per step from the Lagrange basis polynomial over the actual
/// sigma values — <c>∫ Π_{j≠i} (τ − σ_j)/(σ_i − σ_j) dτ</c> across the current step — which is exactly what
/// k-diffusion's <c>linear_multistep_coeff</c> does, and why an LMS run cannot reuse a coefficient table.</para></summary>
public sealed class LmsSampler : ISampler
{
    private const int Order = 4;

    private readonly float[] _sigmas;
    private readonly List<Tensor> _derivatives = [];

    /// <inheritdoc/>
    public string Name => "lms";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public LmsSampler(float[] sigmas)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.Length < 2)
        {
            throw new ArgumentException($"Need at least 2 sigmas; got {sigmas.Length}.", nameof(sigmas));
        }
        _sigmas = sigmas;
    }

    /// <inheritdoc/>
    public void Reset(TensorShape latentShape)
    {
        foreach (Tensor derivative in _derivatives)
        {
            derivative.Dispose();
        }
        _derivatives.Clear();
    }

    /// <inheritdoc/>
    public void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(predictor);
        float sigma = _sigmas[stepIndex];

        using Tensor denoised = SamplerMath.PredictDenoised(backend, predictor, z, sigma, stepIndex);
        Tensor derivative = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, derivative, z, denoised, 1.0f / sigma, -1.0f / sigma);

        _derivatives.Add(derivative);
        if (_derivatives.Count > Order)
        {
            _derivatives[0].Dispose();
            _derivatives.RemoveAt(0);
        }

        int order = Math.Min(stepIndex + 1, Order);
        for (int i = 0; i < order; i++)
        {
            float coefficient = LinearMultistepCoefficient(order, stepIndex, i);
            // _derivatives is oldest-first; coefficient i counts back from the newest.
            SamplerOps.MixInto(backend, z, _derivatives[_derivatives.Count - 1 - i], 1.0f, coefficient);
        }
    }

    /// <summary>Integrates the <paramref name="i"/>-th Lagrange basis polynomial over the current step, giving that
    /// derivative's weight. Simpson's rule on 200 subintervals — the integrand is a smooth low-order polynomial, so
    /// this is far more accuracy than F32 tensors can carry, and it runs once per (step, i) rather than per element.</summary>
    private float LinearMultistepCoefficient(int order, int stepIndex, int i)
    {
        const int Subintervals = 200;
        double lo = _sigmas[stepIndex];
        double hi = _sigmas[stepIndex + 1];
        double width = hi - lo;
        double step = width / Subintervals;
        double total = 0.0;
        for (int k = 0; k <= Subintervals; k++)
        {
            double tau = lo + (k * step);
            double weight = k == 0 || k == Subintervals ? 1.0 : (k % 2 == 1 ? 4.0 : 2.0);
            total += weight * Basis(tau, order, stepIndex, i);
        }
        return (float)(total * step / 3.0);
    }

    /// <summary>The Lagrange basis <c>Π_{j≠i} (τ − σ_{t−j}) / (σ_{t−i} − σ_{t−j})</c>.</summary>
    private double Basis(double tau, int order, int stepIndex, int i)
    {
        double product = 1.0;
        for (int j = 0; j < order; j++)
        {
            if (j == i)
            {
                continue;
            }
            double sigmaI = _sigmas[stepIndex - i];
            double sigmaJ = _sigmas[stepIndex - j];
            product *= (tau - sigmaJ) / (sigmaI - sigmaJ);
        }
        return product;
    }
}
