using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>sa_solver</c>: the stochastic Adams predictor–corrector (predictor order 3, corrector
/// order 4), stochastic between 20% and 80% of the schedule.</summary>
/// <remarks>The pipeline's latent carries the predicted state; the corrected state lives here between steps.</remarks>
public sealed class SaSolverSampler : SamplerBase
{
    private const int PredictorOrder = 3;
    private const int CorrectorOrder = 4;
    private readonly Tensor?[] _predictions = new Tensor?[CorrectorOrder];
    private int _count;
    private Tensor? _corrected;
    private Tensor? _noise;
    private double[] _lambdas = [];
    private double _h;
    private double _tau;
    private double _tauStart;
    private double _tauEnd;
    private bool _lowerOrderToEnd;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public SaSolverSampler(float[] sigmas, int seed, SamplerOptions? options = null)
        : base(sigmas, seed, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "sa_solver";

    /// <inheritdoc/>
    protected override bool OffsetsFirstSigma => true;

    /// <inheritdoc/>
    protected override NoiseKind Noise => NoiseKind.Gaussian;

    /// <inheritdoc/>
    protected override void OnReset()
    {
        for (int k = 0; k < _predictions.Length; k++)
        {
            Release(ref _predictions[k]);
        }
        _count = 0;
        Release(ref _corrected);
        Release(ref _noise);
        _h = 0.0;
        _tau = 0.0;
    }

    /// <inheritdoc/>
    protected override void OnBegin()
    {
        _lambdas = new double[LocalLength];
        for (int k = 0; k < LocalLength; k++)
        {
            _lambdas[k] = Sigma(k) == 0f ? double.PositiveInfinity : Lambda(Sigma(k));
        }
        _tauStart = SigmaAtPercent(0.2);
        _tauEnd = SigmaAtPercent(0.8);
        _lowerOrderToEnd = Sigma(LocalLength - 1) == 0f;
    }

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        float sigma = Sigma(i);
        float sigmaNext = Sigma(i + 1);
        Tensor denoised = Denoise(backend, predictor, z, sigma, stepIndex);
        Release(ref _predictions[^1]);
        for (int k = _predictions.Length - 1; k > 0; k--)
        {
            _predictions[k] = _predictions[k - 1];
        }
        _predictions[0] = null;
        Keep(backend, ref _predictions[0], denoised);
        _count = Math.Min(_count + 1, _predictions.Length);

        int predictorOrder = Math.Min(PredictorOrder, _count);
        int correctorOrder = i == 0 || sigmaNext == 0f ? 0 : Math.Min(CorrectorOrder, _count);
        if (_lowerOrderToEnd)
        {
            predictorOrder = Math.Min(predictorOrder, LocalLength - 2 - i);
            correctorOrder = Math.Min(correctorOrder, LocalLength - 1 - i);
        }

        Span<double> b = stackalloc double[CorrectorOrder];
        if (correctorOrder == 0)
        {
            Tensor copy = new Tensor(z.Shape, DType.F32);
            backend.Scale(copy, z, 1.0f);
            Keep(backend, ref _corrected, copy);
        }
        else
        {
            Coefficients(sigma, i, correctorOrder, _lambdas[i - 1], _lambdas[i], _tau, b);
            Tensor corrected = _corrected!;
            SamplerOps.ScaleInPlace(backend, corrected, (float)(sigma / Sigma(i - 1) * Math.Exp(-(_tau * _tau) * _h)));
            AccumulatePredictions(backend, corrected, correctorOrder, b);
            if (_tau > 0.0 && _noise is not null)
            {
                SamplerOps.MixInto(backend, corrected, _noise, 1.0f, 1.0f);
            }
        }

        if (sigmaNext == 0f)
        {
            backend.Scale(z, _predictions[0]!, 1.0f);
            return;
        }
        _tau = sigmaNext <= _tauStart && sigmaNext >= _tauEnd ? 1.0 : 0.0;
        Coefficients(sigmaNext, i, predictorOrder, _lambdas[i], _lambdas[i + 1], _tau, b);
        _h = _lambdas[i + 1] - _lambdas[i];
        backend.Scale(z, _corrected!, (float)(sigmaNext / sigma * Math.Exp(-(_tau * _tau) * _h)));
        AccumulatePredictions(backend, z, predictorOrder, b);
        if (_tau > 0.0)
        {
            Tensor noise = DrawNoise(stepIndex, 0, sigma, sigmaNext);
            SamplerOps.ScaleInPlace(backend, noise, (float)(sigmaNext * Math.Sqrt(-SamplerMath.Expm1(-2.0 * _tau * _tau * _h))));
            Keep(backend, ref _noise, noise);
            SamplerOps.MixInto(backend, z, noise, 1.0f, 1.0f);
        }
    }

    /// <summary><c>target += Σ b_k · pred_{order−1−k}</c>, oldest first as ComfyUI stacks them.</summary>
    private void AccumulatePredictions(IBackend backend, Tensor target, int order, ReadOnlySpan<double> b)
    {
        for (int k = 0; k < order; k++)
        {
            SamplerOps.MixInto(backend, target, _predictions[order - 1 - k]!, 1.0f, (float)b[k]);
        }
    }

    /// <summary>ComfyUI's <c>compute_stochastic_adams_b_coeffs</c> over <c>lambdas[i−order+1 .. i]</c>.</summary>
    private void Coefficients(double sigmaNext, int i, int order, double lambdaS, double lambdaT, double tau, Span<double> b)
    {
        Span<double> nodes = stackalloc double[CorrectorOrder];
        for (int k = 0; k < order; k++)
        {
            nodes[k] = _lambdas[i - order + 1 + k];
        }
        ComputeCoefficients(sigmaNext, nodes[..order], lambdaS, lambdaT, tau, b[..order]);
    }

    /// <summary>Lagrange-integral weights of the SA-Solver for the given nodes (Vandermonde solve).</summary>
    internal static void ComputeCoefficients(double sigmaNext, ReadOnlySpan<double> nodes, double lambdaS, double lambdaT,
        double tau, Span<double> b)
    {
        int n = nodes.Length;
        double tauMul = 1.0 + (tau * tau);
        double h = lambdaT - lambdaS;
        Span<double> product = stackalloc double[CorrectorOrder];
        for (int p = 0; p < n; p++)
        {
            product[p] = Math.Pow(lambdaT, p) - (Math.Pow(lambdaS, p) * Math.Exp(-tauMul * h));
        }
        Span<double> expCoeffs = stackalloc double[CorrectorOrder];
        for (int a = 0; a < n; a++)
        {
            double sum = 0.0;
            for (int c = 0; c <= a; c++)
            {
                double weight = Factorial(a) / Factorial(c);
                if (tau > 0.0)
                {
                    weight /= Math.Pow(tauMul, a - c);
                }
                sum += ((a - c) % 2 == 0 ? weight : -weight) * product[c];
            }
            expCoeffs[a] = sum;
        }
        Span<double> matrix = stackalloc double[CorrectorOrder * CorrectorOrder];
        for (int p = 0; p < n; p++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[(p * n) + j] = Math.Pow(nodes[j], p);
            }
        }
        Solve(matrix[..(n * n)], expCoeffs[..n], n);
        double alphaT = sigmaNext * Math.Exp(lambdaT);
        for (int k = 0; k < n; k++)
        {
            b[k] = alphaT * expCoeffs[k];
        }
    }

    private static double Factorial(int n)
    {
        double f = 1.0;
        for (int k = 2; k <= n; k++)
        {
            f *= k;
        }
        return f;
    }

    /// <summary>Solves <c>A·x = rhs</c> in place by Gaussian elimination with partial pivoting.</summary>
    private static void Solve(Span<double> a, Span<double> rhs, int n)
    {
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
            {
                if (Math.Abs(a[(r * n) + col]) > Math.Abs(a[(pivot * n) + col]))
                {
                    pivot = r;
                }
            }
            if (pivot != col)
            {
                for (int c = 0; c < n; c++)
                {
                    (a[(col * n) + c], a[(pivot * n) + c]) = (a[(pivot * n) + c], a[(col * n) + c]);
                }
                (rhs[col], rhs[pivot]) = (rhs[pivot], rhs[col]);
            }
            double diag = a[(col * n) + col];
            if (diag == 0.0)
            {
                throw new InvalidOperationException("sa_solver: singular Lagrange system (repeated sigma in the schedule).");
            }
            for (int r = col + 1; r < n; r++)
            {
                double f = a[(r * n) + col] / diag;
                for (int c = col; c < n; c++)
                {
                    a[(r * n) + c] -= f * a[(col * n) + c];
                }
                rhs[r] -= f * rhs[col];
            }
        }
        for (int r = n - 1; r >= 0; r--)
        {
            double sum = rhs[r];
            for (int c = r + 1; c < n; c++)
            {
                sum -= a[(r * n) + c] * rhs[c];
            }
            rhs[r] = sum / a[(r * n) + r];
        }
    }
}
