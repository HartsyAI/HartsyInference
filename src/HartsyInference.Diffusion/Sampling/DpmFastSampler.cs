using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>dpm_fast</c>: DPM-Solver-Fast on its own uniform grid in <c>−log σ</c> from the schedule's
/// first sigma to its smallest non-zero one, spending exactly one model evaluation per schedule step.</summary>
/// <remarks>Solver steps of order k run on the loop index where their k evaluations begin; the loop indices they
/// cover are no-ops. Like the reference, the result is left at the smallest non-zero sigma.</remarks>
public sealed class DpmFastSampler : SamplerBase
{
    private int[] _orders = [];
    private int[] _startIndex = [];
    private double[] _times = [];

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public DpmFastSampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "dpm_fast";

    /// <inheritdoc/>
    protected override void OnBegin()
    {
        int n = LocalLength - 1;
        double sigmaMin = Sigma(n) != 0f ? Sigma(n) : Sigma(n - 1);
        double tStart = -Math.Log(Sigma(0));
        double tEnd = -Math.Log(sigmaMin);
        int m = (n / 3) + 1;
        List<int> orders = [];
        if (n % 3 == 0)
        {
            for (int k = 0; k < m - 2; k++)
            {
                orders.Add(3);
            }
            orders.Add(2);
            orders.Add(1);
        }
        else
        {
            for (int k = 0; k < m - 1; k++)
            {
                orders.Add(3);
            }
            orders.Add(n % 3);
        }
        _orders = [.. orders];
        _times = new double[m + 1];
        for (int k = 0; k <= m; k++)
        {
            _times[k] = tStart + ((tEnd - tStart) * k / m);
        }
        _startIndex = new int[_orders.Length];
        int evaluations = 0;
        for (int k = 0; k < _orders.Length; k++)
        {
            _startIndex[k] = evaluations;
            evaluations += _orders[k];
        }
    }

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        int step = Array.IndexOf(_startIndex, i);
        if (step < 0)
        {
            return;
        }
        double t = _times[step];
        double tNext = _times[step + 1];
        using Tensor eps = DpmSolverSteps.Eps(backend, predictor, z, t, stepIndex);
        using Tensor next = new Tensor(z.Shape, DType.F32);
        Tensor? epsR1 = null;
        try
        {
            switch (_orders[step])
            {
                case 1:
                    DpmSolverSteps.Order1(backend, next, z, eps, t, tNext);
                    break;
                case 2:
                    DpmSolverSteps.Order2(backend, predictor, next, z, eps, t, tNext, 0.5, ref epsR1, stepIndex);
                    break;
                default:
                    DpmSolverSteps.Order3(backend, predictor, next, z, eps, t, tNext, ref epsR1, stepIndex);
                    break;
            }
            backend.Scale(z, next, 1.0f);
        }
        finally
        {
            epsR1?.Dispose();
        }
    }
}
