using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.VibeVoice;

/// <summary>VibeVoice-specific DPM-Solver multistep scheduler. Ports the subset of
/// <c>vibevoice/schedule/dpm_solver.py::DPMSolverMultistepScheduler</c> needed at
/// inference: 1000-step Nichol-Dhariwal cosine beta schedule, v-prediction model output,
/// DPM-Solver++ second-order multistep, 20 inference steps.
///
/// <para>Lives inside <c>SharpInference.Audio</c> so the audio package stays self-contained
/// (the diffusion-pipeline scheduler family lives in <c>SharpInference.Diffusion</c>;
/// adding a cross-package dependency just for one TTS pipeline would muddy the layering).
/// The math is taken verbatim from HuggingFace diffusers DPMSolverMultistepScheduler at
/// <c>algorithm_type="dpmsolver++"</c>, <c>solver_order=2</c>.</para>
///
/// <para>Per-step flow for VibeVoice's diffusion sub-loop:
/// <code>
///   scheduler.SetTimesteps(20)
///   speech = N(0, I) of shape [batch, 64]
///   foreach (t, idx) in (scheduler.Timesteps, indices):
///       v_pred = predictionHead(speech, t, condition)          // v-prediction in two streams (CFG)
///       speech = scheduler.Step(v_pred, t, speech, idx)
///   return speech
/// </code></para></summary>
public sealed class VibeVoiceCosineDpmSolver : IDisposable
{
    private const int NumTrainTimesteps = 1_000;
    private const float CosineSOffset = 0.008f;
    private const float MaxBeta = 0.999f;
    private const float SnrThreshold = -float.MaxValue;     // dpmsolver++ uses every step.

    private readonly float[] _alphaCumprod;     // length NumTrainTimesteps
    private readonly float[] _alphas;           // sqrt(alpha_cumprod)
    private readonly float[] _sigmas;           // sqrt(1 - alpha_cumprod)
    private readonly float[] _lambdas;          // log(alpha / sigma) — the half-noise log-SNR

    private float[] _timesteps = Array.Empty<float>();
    private int _numInferenceSteps;

    // Multistep state — needs the previous v_pred (in "x0 prediction" form).
    private Tensor? _prevX0;
    private float _prevLambda;
    private int _prevStepIndex = -1;

    private int _disposed;

    public int NumInferenceSteps => _numInferenceSteps;
    public ReadOnlySpan<float> Timesteps => _timesteps;

    public VibeVoiceCosineDpmSolver()
    {
        // 1. Nichol-Dhariwal cosine beta schedule.
        //    alpha_bar(t) = cos^2((t/T + s) / (1 + s) * pi/2)
        //    beta[t] = clamp(1 - alpha_bar(t) / alpha_bar(t-1), 0, 0.999)
        float[] betas = new float[NumTrainTimesteps];
        for (int t = 0; t < NumTrainTimesteps; t++)
        {
            double t1 = (double)t / NumTrainTimesteps;
            double t2 = (double)(t + 1) / NumTrainTimesteps;
            double abPrev = AlphaBarCosine(t1);
            double abCur = AlphaBarCosine(t2);
            double beta = 1.0 - abCur / abPrev;
            betas[t] = (float)Math.Min(beta, MaxBeta);
        }

        // 2. alpha_cumprod, then derive alpha (sqrt(alpha_bar)), sigma (sqrt(1-alpha_bar))
        //    and lambda (log-SNR/2).
        _alphaCumprod = new float[NumTrainTimesteps];
        _alphas = new float[NumTrainTimesteps];
        _sigmas = new float[NumTrainTimesteps];
        _lambdas = new float[NumTrainTimesteps];

        double cumprod = 1.0;
        for (int t = 0; t < NumTrainTimesteps; t++)
        {
            cumprod *= (1.0 - betas[t]);
            _alphaCumprod[t] = (float)cumprod;
            _alphas[t] = (float)Math.Sqrt(cumprod);
            _sigmas[t] = (float)Math.Sqrt(1.0 - cumprod);
            _lambdas[t] = (float)(Math.Log(_alphas[t]) - Math.Log(_sigmas[t]));
        }
    }

    /// <summary>Selects <paramref name="num"/> inference timesteps from the 1000-step
    /// training schedule. Uses <c>linspace</c> spacing (matches HF
    /// <c>timestep_spacing="linspace"</c>, the default for DPMSolverMultistepScheduler).</summary>
    public void SetTimesteps(int num)
    {
        if (num <= 0) throw new ArgumentOutOfRangeException(nameof(num));
        _numInferenceSteps = num;
        _timesteps = new float[num];
        // Linspace from NumTrainTimesteps-1 down to 0 (inclusive endpoints).
        float lastStep = NumTrainTimesteps - 1;
        for (int i = 0; i < num; i++)
        {
            float r = (float)i / (num - 1);
            _timesteps[i] = lastStep - r * lastStep;
        }
        _prevX0?.Dispose();
        _prevX0 = null;
        _prevStepIndex = -1;
    }

    /// <summary>Performs one DPM-Solver++ step. <paramref name="vPred"/> is the v-prediction
    /// model output for the current step. <paramref name="sample"/> is the current latent.
    /// Returns the next-step latent (a fresh tensor; caller owns disposal).
    ///
    /// <para>Implements the second-order multistep update. The first step (and any step
    /// after a reset) falls back to a single-step first-order update.</para></summary>
    public unsafe Tensor Step(Tensor vPred, Tensor sample, int stepIndex)
    {
        if (vPred.Shape != sample.Shape)
            throw new ArgumentException($"vPred shape {vPred.Shape} != sample shape {sample.Shape}.");
        if (stepIndex < 0 || stepIndex >= _timesteps.Length)
            throw new ArgumentOutOfRangeException(nameof(stepIndex));

        float t = _timesteps[stepIndex];
        // For v-prediction: x0 = alpha * sample - sigma * v_pred
        (float alpha, float sigma) = InterpolateAlphaSigma(t);
        Tensor x0 = ConvertVPredToX0(vPred, sample, alpha, sigma);

        // Lambda at the current step (interpolated).
        float lambdaT = (float)(Math.Log(alpha) - Math.Log(sigma));

        // Next-step alpha/sigma — clamped at 0 for the final step.
        float nextT = stepIndex + 1 < _timesteps.Length ? _timesteps[stepIndex + 1] : 0f;
        (float alphaNext, float sigmaNext) = InterpolateAlphaSigma(nextT);
        float lambdaNext = nextT > 0f ? (float)(Math.Log(alphaNext) - Math.Log(sigmaNext)) : 1e6f;

        Tensor output;
        if (_prevX0 is null || _prevStepIndex != stepIndex - 1)
        {
            // First-order (multistep init): x_{t-1} = (sigma_{t-1}/sigma_t) * x_t - alpha_{t-1} * (exp(-h)-1) * x0
            // For dpmsolver++ this is:
            //   x_{t-1} = (sigma_next/sigma) * x_t - alpha_next * (exp(-h) - 1) * x0
            // where h = lambda_next - lambda
            float h = lambdaNext - lambdaT;
            output = ApplyFirstOrder(sample, x0, sigmaNext, sigma, alphaNext, h);
        }
        else
        {
            // Second-order multistep with x0 prediction. Uses prevX0 and current x0.
            float h = lambdaNext - lambdaT;
            float hPrev = lambdaT - _prevLambda;
            float r0 = hPrev / h;
            float d1Coeff = 1.0f / r0;
            output = ApplySecondOrder(sample, x0, _prevX0!, sigmaNext, sigma, alphaNext, h, d1Coeff);
        }

        // Update multistep state.
        _prevX0?.Dispose();
        _prevX0 = x0;
        _prevLambda = lambdaT;
        _prevStepIndex = stepIndex;
        return output;
    }

    private static double AlphaBarCosine(double t)
    {
        double f = (t + CosineSOffset) / (1.0 + CosineSOffset) * Math.PI * 0.5;
        double c = Math.Cos(f);
        return c * c;
    }

    /// <summary>Interpolates alpha and sigma at a fractional timestep. Linear interp between
    /// neighbouring integer training-step values matches HF behavior.</summary>
    private (float Alpha, float Sigma) InterpolateAlphaSigma(float t)
    {
        if (t <= 0f) return (1f, 0f);     // perfectly clean at t=0
        if (t >= NumTrainTimesteps - 1) return (_alphas[NumTrainTimesteps - 1], _sigmas[NumTrainTimesteps - 1]);
        int lo = (int)Math.Floor(t);
        int hi = lo + 1;
        float w = t - lo;
        float a = _alphas[lo] * (1f - w) + _alphas[hi] * w;
        float s = _sigmas[lo] * (1f - w) + _sigmas[hi] * w;
        return (a, s);
    }

    /// <summary><c>x0 = alpha_t * sample - sigma_t * v_pred</c> for v-prediction.</summary>
    private static unsafe Tensor ConvertVPredToX0(Tensor vPred, Tensor sample, float alpha, float sigma)
    {
        Tensor x0 = new(sample.Shape, DType.F32);
        long n = sample.ElementCount;
        float* xp = (float*)x0.DataPointer;
        float* vp = (float*)vPred.DataPointer;
        float* sp = (float*)sample.DataPointer;
        for (long i = 0; i < n; i++) xp[i] = alpha * sp[i] - sigma * vp[i];
        return x0;
    }

    /// <summary>First-order DPM-Solver++ step: <c>x_next = (σ_next/σ) · x - α_next · (e^{-h} − 1) · x0</c>.</summary>
    private static unsafe Tensor ApplyFirstOrder(Tensor sample, Tensor x0, float sigmaNext, float sigma, float alphaNext, float h)
    {
        Tensor output = new(sample.Shape, DType.F32);
        long n = sample.ElementCount;
        float* op = (float*)output.DataPointer;
        float* sp = (float*)sample.DataPointer;
        float* xp = (float*)x0.DataPointer;
        float ratio = sigmaNext / sigma;
        // (exp(-h) - 1) factor in dpmsolver++ for x0.
        float coef = alphaNext * (MathF.Exp(-h) - 1f);
        for (long i = 0; i < n; i++) op[i] = ratio * sp[i] - coef * xp[i];
        return output;
    }

    /// <summary>Second-order DPM-Solver++ multistep step. Adds a finite-difference correction
    /// using the previous x0 prediction:
    /// <code>
    ///   d1 = (x0 - x0_prev) / r0
    ///   x_next = (σ_next/σ) · x - α_next · (e^{-h} − 1) · x0 - 0.5 · α_next · (e^{-h} − 1) · d1
    /// </code></summary>
    private static unsafe Tensor ApplySecondOrder(Tensor sample, Tensor x0, Tensor x0Prev,
        float sigmaNext, float sigma, float alphaNext, float h, float d1Coeff)
    {
        Tensor output = new(sample.Shape, DType.F32);
        long n = sample.ElementCount;
        float* op = (float*)output.DataPointer;
        float* sp = (float*)sample.DataPointer;
        float* xp = (float*)x0.DataPointer;
        float* pp = (float*)x0Prev.DataPointer;
        float ratio = sigmaNext / sigma;
        float coef = alphaNext * (MathF.Exp(-h) - 1f);
        float halfCoef = 0.5f * coef;
        for (long i = 0; i < n; i++)
        {
            float d1 = (xp[i] - pp[i]) * d1Coeff;
            op[i] = ratio * sp[i] - coef * xp[i] - halfCoef * d1;
        }
        return output;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _prevX0?.Dispose();
        _prevX0 = null;
    }
}
