using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>DPMSolver</c> single steps in eps form, shared by <c>dpm_fast</c> and <c>dpm_adaptive</c>.</summary>
/// <remarks>Time is <c>t = −log σ</c>. Each step reuses the caller's eps at <c>t</c>, as the reference's eps cache does.</remarks>
internal static class DpmSolverSteps
{
    /// <summary><c>(x − D(x, σ))/σ</c> at <c>σ = e^−t</c>; the caller owns the result.</summary>
    public static Tensor Eps(IBackend backend, IDenoisePredictor predictor, Tensor x, double t, int stepIndex)
    {
        float sigma = (float)Math.Exp(-t);
        using Tensor denoised = SamplerMath.PredictDenoised(backend, predictor, x, sigma, stepIndex);
        Tensor eps = new Tensor(x.Shape, DType.F32);
        SamplerOps.SetMix(backend, eps, x, denoised, 1.0f / sigma, -1.0f / sigma);
        return eps;
    }

    /// <summary>DPM-Solver-1 into <paramref name="output"/>.</summary>
    public static void Order1(IBackend backend, Tensor output, Tensor x, Tensor eps, double t, double tNext)
    {
        double h = tNext - t;
        SamplerOps.SetMix(backend, output, x, eps, 1.0f, (float)(-Math.Exp(-tNext) * SamplerMath.Expm1(h)));
    }

    /// <summary>DPM-Solver-2 into <paramref name="output"/>; <paramref name="epsR1"/> is computed on demand and returned.</summary>
    public static void Order2(IBackend backend, IDenoisePredictor predictor, Tensor output, Tensor x, Tensor eps, double t,
        double tNext, double r1, ref Tensor? epsR1, int stepIndex)
    {
        double h = tNext - t;
        double s1 = t + (r1 * h);
        if (epsR1 is null)
        {
            using Tensor u1 = new Tensor(x.Shape, DType.F32);
            SamplerOps.SetMix(backend, u1, x, eps, 1.0f, (float)(-Math.Exp(-s1) * SamplerMath.Expm1(r1 * h)));
            epsR1 = Eps(backend, predictor, u1, s1, stepIndex);
        }
        double sigmaNext = Math.Exp(-tNext);
        double a = -sigmaNext * SamplerMath.Expm1(h);
        double c = -sigmaNext / (2.0 * r1) * SamplerMath.Expm1(h);
        SamplerOps.SetMix(backend, output, x, eps, epsR1, 1.0f, (float)(a - c), (float)c);
    }

    /// <summary>DPM-Solver-3 (r1 = 1/3, r2 = 2/3) into <paramref name="output"/>, reusing <paramref name="epsR1"/>.</summary>
    public static void Order3(IBackend backend, IDenoisePredictor predictor, Tensor output, Tensor x, Tensor eps, double t,
        double tNext, ref Tensor? epsR1, int stepIndex)
    {
        const double R1 = 1.0 / 3.0;
        const double R2 = 2.0 / 3.0;
        double h = tNext - t;
        double s1 = t + (R1 * h);
        double s2 = t + (R2 * h);
        if (epsR1 is null)
        {
            using Tensor u1 = new Tensor(x.Shape, DType.F32);
            SamplerOps.SetMix(backend, u1, x, eps, 1.0f, (float)(-Math.Exp(-s1) * SamplerMath.Expm1(R1 * h)));
            epsR1 = Eps(backend, predictor, u1, s1, stepIndex);
        }
        double sigmaS2 = Math.Exp(-s2);
        double a2 = -sigmaS2 * SamplerMath.Expm1(R2 * h);
        double c2 = -sigmaS2 * (R2 / R1) * ((SamplerMath.Expm1(R2 * h) / (R2 * h)) - 1.0);
        using Tensor u2 = new Tensor(x.Shape, DType.F32);
        SamplerOps.SetMix(backend, u2, x, eps, epsR1, 1.0f, (float)(a2 - c2), (float)c2);
        using Tensor epsR2 = Eps(backend, predictor, u2, s2, stepIndex);
        double sigmaNext = Math.Exp(-tNext);
        double a = -sigmaNext * SamplerMath.Expm1(h);
        double c = -sigmaNext / R2 * ((SamplerMath.Expm1(h) / h) - 1.0);
        SamplerOps.SetMix(backend, output, x, eps, epsR2, 1.0f, (float)(a - c), (float)c);
    }
}
