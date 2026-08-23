using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using RefinerRequest = HartsyInference.Engine.Requests.Refiner;

namespace HartsyInference.Engine.Features;

/// <summary>Normalizes the request's <see cref="RefinerRequest"/> into a concrete second-pass plan, filling the toggleable step/CFG overrides from the base pass when unset. Returns null when no refiner stage should run (no model, or a control fraction of 0).</summary>
public static class RefinerResolver
{
    /// <summary>A resolved refiner pass: which checkpoint, how many steps of it, and at what strength/CFG.</summary>
    public sealed class RefinerSpec
    {
        /// <summary>Refiner model id or path.</summary>
        public required string Model { get; init; }

        /// <summary>Refiner VAE override; null reuses the base VAE.</summary>
        public string? Vae { get; init; }

        /// <summary>Step total used for the refiner sub-range calculation.</summary>
        public required int Steps { get; init; }

        /// <summary>Fraction of the schedule the refiner runs (the request's <c>Control</c>).</summary>
        public required float Strength { get; init; }

        /// <summary>CFG for the refiner pass.</summary>
        public required float CfgScale { get; init; }

        /// <summary>Latent upscale factor applied before the refiner pass (1 = none).</summary>
        public required float Upscale { get; init; }

        /// <summary>Hand-off method: <c>"StepSwap"</c> (mid-loop UNet swap, same resolution throughout) or <c>"PostApply"</c> (full pixel-space roundtrip, the resolver's own default below — the only one that can change resolution, Tier 3.1 hires-fix). Both are wired in <see cref="Recipes.Image.SdxlRecipePipeline"/>.</summary>
        public required string Method { get; init; }
    }

    /// <summary>Resolves the refiner pass against the base pass's <paramref name="baseSteps"/> / <paramref name="baseCfg"/>.</summary>
    public static RefinerSpec? Resolve(RefinerRequest? refiner, int baseSteps, float baseCfg)
    {
        if (refiner is null || string.IsNullOrWhiteSpace(refiner.Model))
        {
            return null;
        }
        double control = refiner.Control ?? 0.0;
        if (control <= 0)
        {
            Logs.Debug("[Features][Refiner] Control=0 → skipping refiner pass.");
            return null;
        }
        // Steps is toggleable: only honored when explicitly set.
        int refinerStepTotal = refiner.Steps is > 0 ? refiner.Steps.Value : baseSteps;
        if (refinerStepTotal <= 0)
        {
            refinerStepTotal = 30;
        }
        // CFG is also toggleable — fall back to the base CFG when off.
        double cfg = refiner.CfgScale ?? baseCfg;
        return new RefinerSpec
        {
            Model = refiner.Model,
            Vae = refiner.Vae,
            Steps = refinerStepTotal,
            Strength = (float)control,
            CfgScale = cfg <= 0 ? 7.5f : (float)cfg,
            Upscale = refiner.Upscale is > 0 ? (float)refiner.Upscale.Value : 1f,
            Method = string.IsNullOrWhiteSpace(refiner.Method) ? "PostApply" : refiner.Method,
        };
    }
}
