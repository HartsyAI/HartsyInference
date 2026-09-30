using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>dpm_adaptive</c>: DPM-Solver-23 with PID step-size control from the schedule's first sigma
/// to its smallest non-zero one.</summary>
/// <remarks>The step count is chosen by the error controller, so the whole solve runs on the first loop step and the
/// remaining steps are no-ops. The error norm is read back to the host once per attempt.</remarks>
public sealed class DpmAdaptiveSampler : SamplerBase
{
    private const double RelativeTolerance = 0.05;
    private const double AbsoluteTolerance = 0.0078;
    private const double InitialStep = 0.05;
    private const double AcceptSafety = 0.81;
    private const double ControllerExponent = 1.0 / 3.0;
    private const int MaxAttempts = 10000;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public DpmAdaptiveSampler(float[] sigmas, SamplerOptions? options = null)
        : base(sigmas, 0, options)
    {
    }

    /// <inheritdoc/>
    public override string Name => "dpm_adaptive";

    /// <summary>Model evaluations the last run took.</summary>
    public int LastEvaluations { get; private set; }

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        if (i != 0)
        {
            return;
        }
        int n = LocalLength - 1;
        double sigmaMin = Sigma(n) != 0f ? Sigma(n) : Sigma(n - 1);
        double tEnd = -Math.Log(sigmaMin);
        double s = -Math.Log(Sigma(0));
        double h = InitialStep;
        double[] errors = [];
        int evaluations = 0;
        using Tensor previous = new Tensor(z.Shape, DType.F32);
        backend.Scale(previous, z, 1.0f);
        using Tensor low = new Tensor(z.Shape, DType.F32);
        using Tensor high = new Tensor(z.Shape, DType.F32);
        for (int attempt = 0; s < tEnd - 1e-5; attempt++)
        {
            if (attempt >= MaxAttempts)
            {
                throw new InvalidOperationException($"dpm_adaptive did not reach sigma {sigmaMin} within {MaxAttempts} attempts.");
            }
            double t = Math.Min(tEnd, s + h);
            Tensor? epsR1 = null;
            try
            {
                using Tensor eps = DpmSolverSteps.Eps(backend, predictor, z, s, stepIndex);
                DpmSolverSteps.Order2(backend, predictor, low, z, eps, s, t, 1.0 / 3.0, ref epsR1, stepIndex);
                DpmSolverSteps.Order3(backend, predictor, high, z, eps, s, t, ref epsR1, stepIndex);
            }
            finally
            {
                epsR1?.Dispose();
            }
            evaluations += 3;
            double error = ErrorNorm(low, high, previous);
            double inverse = 1.0 / (error + 1e-8);
            if (errors.Length == 0)
            {
                errors = [inverse, inverse, inverse];
            }
            errors[0] = inverse;
            double factor = 1.0 + Math.Atan(Math.Pow(errors[0], ControllerExponent) - 1.0);
            if (!double.IsFinite(factor))
            {
                throw new InvalidOperationException("dpm_adaptive step-size control produced a non-finite error; the model output is not finite.");
            }
            h *= factor;
            if (factor >= AcceptSafety)
            {
                errors[2] = errors[1];
                errors[1] = errors[0];
                backend.Scale(previous, low, 1.0f);
                backend.Scale(z, high, 1.0f);
                s = t;
            }
        }
        LastEvaluations = evaluations;
    }

    /// <summary><c>‖(low − high)/max(atol, rtol·max(|low|, |previous|))‖₂ / √numel</c>, on the host.</summary>
    private static unsafe double ErrorNorm(Tensor low, Tensor high, Tensor previous)
    {
        float* pl = (float*)low.DataPointer;
        float* ph = (float*)high.DataPointer;
        float* pp = (float*)previous.DataPointer;
        long count = low.Shape.ElementCount;
        double sum = 0.0;
        for (long k = 0; k < count; k++)
        {
            double delta = Math.Max(AbsoluteTolerance, RelativeTolerance * Math.Max(Math.Abs(pl[k]), Math.Abs(pp[k])));
            double e = (pl[k] - ph[k]) / delta;
            sum += e * e;
        }
        return Math.Sqrt(sum) / Math.Sqrt(count);
    }
}
