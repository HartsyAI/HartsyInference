using HartsyInference.Core.Backends.Guidance;
using System.Threading;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>Rectified-flow Euler sampler of the HairDeck set DiTs (<c>hairdeck/sampling/flow.py</c>): integrates from the
/// noise at t = 1 to the data at t = 0 with <c>v = v_uncond + cfg (v_cond - v_uncond)</c>.</summary>
public static class HairDeckEulerSampler
{
    /// <summary>Samples tokens from the initial noise <paramref name="noise"/> <c>[N, TokenDim]</c>; the caller disposes the result.</summary>
    /// <param name="onStep">Called with the number of steps done after each step.</param>
    public static Tensor Sample(
        IBackend backend, HairDeckSetDit model, Tensor ctx, Tensor noise, int steps, float cfg, CancellationToken cancel = default, Action<int>? onStep = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "The sampler needs at least one step.");
        if (!float.IsFinite(cfg))
            throw new ArgumentOutOfRangeException(nameof(cfg), cfg, "The guidance scale must be finite.");

        Tensor x = HairDeckNumerics.Clone(noise);
        int points = steps + 1;
        for (int i = 0; i < steps; i++)
        {
            cancel.ThrowIfCancellationRequested();
            float t = HairDeckNumerics.Linspace(i, points, 1f, 0f);
            float tNext = HairDeckNumerics.Linspace(i + 1, points, 1f, 0f);
            float delta = tNext - t;
            using Tensor vCond = model.Forward(backend, x, t, ctx, uncond: false);
            if (cfg == 1f)
            {
                backend.GuidedEulerStep(x, vCond, vCond, GuidedEulerSpec.Lerp(1f, delta));
            }
            else
            {
                using Tensor vUncond = model.Forward(backend, x, t, ctx, uncond: true);
                backend.GuidedEulerStep(x, vCond, vUncond, GuidedEulerSpec.Lerp(cfg, delta));
            }
            onStep?.Invoke(i + 1);
        }
        return x;
    }
}
