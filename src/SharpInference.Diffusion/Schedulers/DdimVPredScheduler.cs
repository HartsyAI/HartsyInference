using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Schedulers;

/// <summary>DDIM with v-parameterization on the sigmoid β-schedule (T=1000, start=−3, end=3) — the scheduler family
/// Oasis-500m uses, distinct from the flow-matching stack. Built for <b>Diffusion Forcing</b> inference: each frame in
/// the window carries its own integer noise index (context frames sit at a fixed stabilization level while the target
/// frame walks the DDIM range), so the step API operates on one frame at a time with explicit <c>t → t_next</c>
/// indices. <c>t = −1</c> means clean (<c>ᾱ = 1</c>). See <c>docs/Research/OASIS_ARCHITECTURE.md</c> § 7.</summary>
public sealed unsafe class DdimVPredScheduler
{
    private readonly float[] _alphasCumprod;

    public DdimVPredScheduler(int numTrainTimesteps = 1000, double start = -3, double end = 3, double tau = 1)
    {
        float[] betas = SigmoidBetaSchedule(numTrainTimesteps, start, end, tau);
        _alphasCumprod = new float[numTrainTimesteps];
        double acc = 1;
        for (int i = 0; i < numTrainTimesteps; i++)
        {
            acc *= 1 - betas[i];
            _alphasCumprod[i] = (float)acc;
        }
    }

    /// <summary>Training-resolution ᾱ table (length T).</summary>
    public ReadOnlySpan<float> AlphasCumprod => _alphasCumprod;

    /// <summary>ᾱ at noise index <paramref name="t"/>; <c>t &lt; 0</c> is the clean state (ᾱ = 1).</summary>
    public float AlphaCumprod(int t) => t < 0 ? 1f : _alphasCumprod[t];

    /// <summary>The sigmoid β-schedule (Chen 2022 Fig. 8): <c>ᾱ(t) = (σ(end) − σ((t·(end−start)+start)/τ)) /
    /// (σ(end) − σ(start))</c> normalized to 1 at t=0, returned as per-step betas clamped to [0, 0.999].</summary>
    public static float[] SigmoidBetaSchedule(int timesteps, double start = -3, double end = 3, double tau = 1)
    {
        double vStart = Sigmoid(start / tau);
        double vEnd = Sigmoid(end / tau);
        double[] alphasCumprod = new double[timesteps + 1];
        for (int i = 0; i <= timesteps; i++)
        {
            double t = (double)i / timesteps;
            alphasCumprod[i] = (-Sigmoid((t * (end - start) + start) / tau) + vEnd) / (vEnd - vStart);
        }
        double a0 = alphasCumprod[0];
        float[] betas = new float[timesteps];
        for (int i = 0; i < timesteps; i++)
            betas[i] = (float)Math.Clamp(1 - (alphasCumprod[i + 1] / a0) / (alphasCumprod[i] / a0), 0, 0.999);
        return betas;
    }

    /// <summary>The DDIM noise-index ladder <c>linspace(−1, T−1, steps+1)</c> — Oasis iterates indices
    /// <c>steps … 1</c>, stepping each frame from <c>range[i]</c> to <c>range[i−1]</c> (the terminal −1 lands clean).</summary>
    public int[] BuildNoiseRange(int ddimSteps)
    {
        int[] range = new int[ddimSteps + 1];
        int t = _alphasCumprod.Length;
        for (int i = 0; i <= ddimSteps; i++)
            range[i] = (int)Math.Round(-1 + (double)(t - 1 - (-1)) * i / ddimSteps);
        return range;
    }

    /// <summary>One v-parameterization DDIM update, in place on <paramref name="frame"/>:
    /// <c>x₀ = √ᾱ_t·x − √(1−ᾱ_t)·v</c>, <c>ε = (√(1/ᾱ_t)·x − x₀)/√(1/ᾱ_t − 1)</c>,
    /// <c>x ← √ᾱ_next·x₀ + √(1−ᾱ_next)·ε</c>. With <paramref name="tNext"/> &lt; 0 the frame lands exactly on x₀.</summary>
    public void StepFrame(Tensor frame, Tensor v, int t, int tNext)
    {
        if (t < 0) throw new ArgumentOutOfRangeException(nameof(t), "current noise index must be ≥ 0.");
        if (frame.Shape.ElementCount != v.Shape.ElementCount)
            throw new ArgumentException("v element count must match the frame.", nameof(v));

        float alphaT = AlphaCumprod(t);
        float alphaNext = AlphaCumprod(tNext);
        float sqrtAlphaT = MathF.Sqrt(alphaT);
        float sqrtOneMinusAlphaT = MathF.Sqrt(1f - alphaT);
        float sqrtInvAlphaT = MathF.Sqrt(1f / alphaT);
        float invNoiseScale = 1f / MathF.Sqrt(1f / alphaT - 1f);
        float sqrtAlphaNext = MathF.Sqrt(alphaNext);
        float sqrtOneMinusAlphaNext = MathF.Sqrt(Math.Max(0f, 1f - alphaNext));

        float* xp = (float*)frame.DataPointer;
        float* vp = (float*)v.DataPointer;
        long n = frame.Shape.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float x = xp[i];
            float xStart = sqrtAlphaT * x - sqrtOneMinusAlphaT * vp[i];
            float xNoise = (sqrtInvAlphaT * x - xStart) * invNoiseScale;
            xp[i] = sqrtAlphaNext * xStart + sqrtOneMinusAlphaNext * xNoise;
        }
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
}
