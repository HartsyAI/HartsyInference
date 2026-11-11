using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// SAM 3D's Euler sampler over a named set of latents: the data sits at <c>t = 1</c> and the noise at <c>t = 0</c>, the grid is
/// <c>linspace(0, 1, steps + 1)</c> warped by <c>t / (1 + (rescale_t − 1)(1 − t))</c>, and the model sees <c>time_scale · t</c>.
/// Guidance is <c>(1 + s)·cond − s·uncond</c> on every latent with one scalar strength, active while the scaled time is inside the
/// interval; outside it only the conditional pass runs. The conditional and unconditional passes differ only in the zeroed condition.
/// </summary>
public static class SamFlowSampler
{
    /// <summary>Model velocities per latent for <c>(latents, scaled t, scaled d, unconditional)</c>; the sampler disposes the result, so every tensor must be a distinct object.</summary>
    public delegate IReadOnlyDictionary<string, Tensor> Predict(
        IReadOnlyDictionary<string, Tensor> x, float tModel, float dModel, bool unconditional);

    /// <summary>The warped time grid, ascending from 0 to 1.</summary>
    public static double[] Schedule(int steps, double rescaleT)
    {
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "Steps must be positive.");
        double[] t = new double[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            double linear = i == steps ? 1.0 : (double)i / steps;
            t[i] = linear / (1.0 + (rescaleT - 1.0) * (1.0 - linear));
        }

        return t;
    }

    /// <summary>True when step time <paramref name="t"/> is guided: <c>low &lt;= time_scale·t &lt;= high</c> and the strength is positive.</summary>
    public static bool IsGuided(double t, SamSamplerParams p) =>
        p.Strength > 0f && p.IntervalLow <= p.TimeScale * t && p.TimeScale * t <= p.IntervalHigh;

    /// <summary>Integrates the noise latents to data; the caller owns the returned tensors and the noise is not modified.</summary>
    /// <param name="onStep">Called after each step; it may throw to cancel, and the latents are released first.</param>
    public static Dictionary<string, Tensor> Sample(
        IBackend backend, IReadOnlyDictionary<string, Tensor> noise, SamSamplerParams parameters, Predict predict, Action<int>? onStep = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(predict);
        parameters.Validate();
        if (noise.Count == 0)
            throw new ArgumentException("At least one latent is required.", nameof(noise));

        double[] t = Schedule(parameters.Steps, parameters.RescaleT);
        Dictionary<string, Tensor> x = new(noise.Count);
        try
        {
            foreach (KeyValuePair<string, Tensor> entry in noise)
            {
                if (entry.Value.DType != DType.F32)
                    throw new NotSupportedException($"The sampler runs in F32; latent '{entry.Key}' is {entry.Value.DType}.");
                Tensor copy = new(entry.Value.Shape, DType.F32);
                x.Add(entry.Key, copy);
                backend.CopyInto(copy, entry.Value);
            }

            for (int step = 0; step < parameters.Steps; step++)
            {
                Step(backend, x, t[step], t[step + 1], parameters, predict);
                onStep?.Invoke(step + 1);
            }

            return x;
        }
        catch
        {
            foreach (Tensor tensor in x.Values)
                tensor.Dispose();
            throw;
        }
    }

    private static void Step(IBackend backend, Dictionary<string, Tensor> x, double t, double tNext, SamSamplerParams p, Predict predict)
    {
        float tModel = (float)(p.TimeScale * t);
        float dModel = (float)p.ShortcutStep;
        float delta = (float)(tNext - t);
        IReadOnlyDictionary<string, Tensor> latents = x;
        if (!IsGuided(t, p))
        {
            IReadOnlyDictionary<string, Tensor> v = predict(latents, tModel, dModel, false);
            try
            {
                foreach (KeyValuePair<string, Tensor> entry in x)
                    backend.GuidedEulerStep(entry.Value, Velocity(entry.Key, entry.Value, v), Velocity(entry.Key, entry.Value, v), GuidedEulerSpec.Lerp(1f, delta));
            }
            finally
            {
                Release(v);
            }

            return;
        }

        IReadOnlyDictionary<string, Tensor> cond = predict(latents, tModel, dModel, false);
        try
        {
            IReadOnlyDictionary<string, Tensor> uncond = predict(latents, tModel, dModel, true);
            try
            {
                // z += ((1 + s)·cond − s·uncond)·dt is the Lerp form with guidance 1 + s.
                GuidedEulerSpec spec = GuidedEulerSpec.Lerp(1f + p.Strength, delta);
                foreach (KeyValuePair<string, Tensor> entry in x)
                    backend.GuidedEulerStep(entry.Value, Velocity(entry.Key, entry.Value, cond), Velocity(entry.Key, entry.Value, uncond), spec);
            }
            finally
            {
                Release(uncond);
            }
        }
        finally
        {
            Release(cond);
        }
    }

    private static Tensor Velocity(string name, Tensor latent, IReadOnlyDictionary<string, Tensor> velocities)
    {
        if (!velocities.TryGetValue(name, out Tensor? v))
            throw new InvalidOperationException($"The model returned no velocity for latent '{name}'.");
        if (v.DType != DType.F32 || v.Shape != latent.Shape)
            throw new InvalidOperationException($"The model returned {v.DType} {v.Shape} for latent '{name}' of {latent.Shape}.");
        return v;
    }

    private static void Release(IReadOnlyDictionary<string, Tensor> velocities)
    {
        foreach (Tensor tensor in velocities.Values)
            tensor.Dispose();
    }
}
