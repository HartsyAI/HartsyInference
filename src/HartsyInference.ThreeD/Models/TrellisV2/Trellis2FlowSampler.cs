using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Upstream <c>FlowEulerGuidanceIntervalSampler</c>: Euler integration of the rectified flow from <c>t = 1</c> (noise) to 0 on a
/// time-warped schedule, with classifier-free guidance <c>s·pos + (1 − s)·neg</c> inside the guidance interval and an optional
/// std-matched rescale of the guided <c>x0</c>. Layout-agnostic: it only does element-wise math over the whole latent, so the
/// dense stage-1 latent and the sparse shape latent share it. The model sees <c>1000 · t</c>.
/// </summary>
public static class Trellis2FlowSampler
{
    /// <summary>Velocity of the model for the current latent: <c>(x, 1000·t, conditional)</c>; the sampler disposes the result.</summary>
    public delegate Tensor Predict(Tensor x, float tModel, bool conditional);

    /// <summary>The time grid <c>rescale_t·t / (1 + (rescale_t − 1)·t)</c> over <c>linspace(1, 0, steps + 1)</c>.</summary>
    public static double[] Schedule(int steps, double rescaleT)
    {
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "Steps must be positive.");
        double[] t = new double[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            double linear = i == steps ? 0.0 : 1.0 + i * (-1.0 / steps);
            t[i] = rescaleT * linear / (1.0 + (rescaleT - 1.0) * linear);
        }

        return t;
    }

    /// <summary>Denoises <paramref name="noise"/> (F32, any shape); the caller owns the returned sample, and the noise is not modified.</summary>
    /// <param name="onStep">Called after each step; it may throw to cancel, and the sample is released first.</param>
    public static Tensor Sample(IBackend backend, Tensor noise, Trellis2SamplerParams parameters, Predict predict, Action<int>? onStep = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(predict);
        parameters.Validate();
        if (noise.DType != DType.F32)
            throw new NotSupportedException($"The sampler runs in F32; the noise is {noise.DType}.");

        double[] t = Schedule(parameters.Steps, parameters.RescaleT);
        Tensor x = new(noise.Shape, DType.F32);
        backend.CopyInto(x, noise);
        try
        {
            for (int step = 0; step < parameters.Steps; step++)
            {
                Step(backend, x, t[step], t[step + 1], parameters, predict);
                onStep?.Invoke(step + 1);
            }

            return x;
        }
        catch
        {
            x.Dispose();
            throw;
        }
    }

    private static void Step(IBackend backend, Tensor x, double t, double tPrev, Trellis2SamplerParams p, Predict predict)
    {
        float tModel = (float)(1000.0 * t);
        float delta = (float)(tPrev - t);
        bool guided = p.IntervalLow <= t && t <= p.IntervalHigh;
        float strength = guided ? p.GuidanceStrength : 1f;
        if (strength == 1f || strength == 0f)
        {
            // One model call: the conditional prediction, or the unconditional one when the strength is exactly 0.
            using Tensor v = predict(x, tModel, strength == 1f);
            RequireVelocity(x, v);
            backend.GuidedEulerStep(x, v, v, GuidedEulerSpec.Lerp(1f, delta));
            return;
        }

        using Tensor pos = predict(x, tModel, true);
        RequireVelocity(x, pos);
        using Tensor neg = predict(x, tModel, false);
        RequireVelocity(x, neg);
        if (p.GuidanceRescale > 0f)
        {
            RescaleGuidance(x, pos, neg, strength, p.GuidanceRescale, t, p.SigmaMin);
            // pos now holds the rescaled guided prediction.
            backend.GuidedEulerStep(x, pos, pos, GuidedEulerSpec.Lerp(1f, delta));
            return;
        }

        backend.GuidedEulerStep(x, pos, neg, GuidedEulerSpec.Lerp(strength, delta));
    }

    /// <summary>
    /// Replaces <paramref name="pos"/> with the guided prediction after upstream's CFG rescale: <c>x0</c> is formed from the
    /// conditional and the guided prediction, the guided one is scaled by <c>std(x0_pos) / std(x0_cfg)</c> and blended with weight
    /// <paramref name="rescale"/>, and the blend is converted back to a prediction. The std runs over every element of the
    /// latent. Upstream's dense path uses the unbiased std and its sparse path the biased one; the ratio is the same.
    /// </summary>
    internal static void RescaleGuidance(Tensor x, Tensor pos, Tensor neg, float strength, float rescale, double t, double sigmaMin)
        => RescaleGuidance(x.AsReadOnlySpan<float>(), pos.AsSpan<float>(), neg.AsReadOnlySpan<float>(), strength, rescale, t, sigmaMin);

    internal static void RescaleGuidance(
        ReadOnlySpan<float> x, Span<float> pos, ReadOnlySpan<float> neg, float strength, float rescale, double t, double sigmaMin)
    {
        double a = 1.0 - sigmaMin;
        double b = sigmaMin + a * t;
        double meanPos = 0.0, meanCfg = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
            double guided = strength * (double)pos[i] + (1.0 - strength) * neg[i];
            meanPos += a * x[i] - b * pos[i];
            meanCfg += a * x[i] - b * guided;
        }

        meanPos /= x.Length;
        meanCfg /= x.Length;
        double varPos = 0.0, varCfg = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
            double guided = strength * (double)pos[i] + (1.0 - strength) * neg[i];
            double dp = a * x[i] - b * pos[i] - meanPos;
            double dc = a * x[i] - b * guided - meanCfg;
            varPos += dp * dp;
            varCfg += dc * dc;
        }

        // A guided x0 with no spread has nothing to match; keep the plain guided prediction.
        double ratio = varCfg > 0.0 ? Math.Sqrt(varPos / varCfg) : 1.0;
        double blend = rescale * ratio + (1.0 - rescale);
        for (int i = 0; i < x.Length; i++)
        {
            double guided = strength * (double)pos[i] + (1.0 - strength) * neg[i];
            double x0 = (a * x[i] - b * guided) * blend;
            pos[i] = (float)((a * x[i] - x0) / b);
        }
    }

    private static void RequireVelocity(Tensor x, Tensor v)
    {
        if (v.DType != DType.F32 || v.Shape.ElementCount != x.Shape.ElementCount)
            throw new InvalidOperationException($"The model returned {v.DType} {v.Shape} for a latent of {x.Shape}.");
    }
}
