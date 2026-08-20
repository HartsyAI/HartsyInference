using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Recipes;

/// <summary>Says out loud when a recipe cannot honor a configured placement or low-VRAM setting.</summary>
/// <remarks>Every knob here is configured somewhere other than the recipe — a second backend built by
/// <see cref="RecipeContext.AllBackends"/>, an environment variable, a host-side override — so a recipe that simply
/// does not consume one produces a working generation that silently ignores it. Wan-Animate ignored all four, and
/// diagnosing that from the outside cost a full session: the operator sees a setting take effect nowhere and has no
/// way to tell "not wired" from "wired and not helping".</remarks>
public static class PlacementSupport
{
    /// <summary>Warns for each configured capability <paramref name="recipeName"/> does not implement, and notes any
    /// component backend placed off the primary.</summary>
    /// <param name="contextParallel">Whether the recipe consumes <see cref="RecipeContext.CpBackends"/>.</param>
    /// <param name="cfgParallel">Whether the recipe consumes <see cref="RecipeContext.CfgParallelBackend"/>.</param>
    /// <param name="ditSharding">Whether the recipe consumes <see cref="RecipeContext.DitShardBackend"/>.</param>
    /// <param name="blockStreaming">Whether the recipe's denoiser opts into <see cref="IStreamableDenoiser"/>.</param>
    public static void WarnIfIgnored(string recipeName, RecipeContext context, bool contextParallel = false,
        bool cfgParallel = false, bool ditSharding = false, bool blockStreaming = false)
    {
        ArgumentNullException.ThrowIfNull(recipeName);
        ArgumentNullException.ThrowIfNull(context);
        if (!contextParallel && context.CpBackends is { Count: > 0 })
        {
            Logs.Warning($"[{recipeName}] Context parallelism is configured but not wired for this model — the "
                + "generation runs single-GPU and the extra backend is unused.");
        }
        if (!cfgParallel && context.CfgParallelBackend is not null)
        {
            Logs.Warning($"[{recipeName}] CFG-parallel is configured but not wired for this model — the "
                + "unconditional branch runs sequentially on the primary backend.");
        }
        if (!ditSharding && (context.DitShardBackend is not null || context.DitShardBackends is { Count: > 0 }))
        {
            Logs.Warning($"[{recipeName}] DiT sharding is configured but not wired for this model — the whole "
                + "denoiser runs on the primary backend.");
        }
        // Only ForceOn is worth a warning: it is the setting an operator turns on and then watches for an effect.
        // Auto is the default on every backend, and warning about it would fire on every generation everywhere.
        if (!blockStreaming && LowVramPolicy.Resolve(context.Backend) == LowVramMode.ForceOn)
        {
            Logs.Warning($"[{recipeName}] Low-VRAM streaming is requested but this model's denoiser exposes no "
                + "streamable blocks — every weight stays resident, so an oversized request will still fail.");
        }
        WarnIfComponentsSplit(recipeName, context);
    }

    /// <summary>Notes text-encoder / VAE placement off the primary backend — the one placement setting whose effect
    /// is invisible in the log because it makes things work rather than fail.</summary>
    public static void WarnIfComponentsSplit(string recipeName, RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(recipeName);
        ArgumentNullException.ThrowIfNull(context);
        if (context.TextEncoderBackend is not null && !ReferenceEquals(context.TextEncoderBackend, context.Backend))
        {
            Logs.Info($"[{recipeName}] Text encoder placed off the primary backend — its weights stay resident "
                + "between generations, since nothing on that device competes with the denoiser for the space.");
        }
        if (context.VaeBackend is not null && !ReferenceEquals(context.VaeBackend, context.Backend))
        {
            Logs.Info($"[{recipeName}] VAE encode/decode placed off the primary backend — its weights stay "
                + "resident between generations, since nothing on that device competes with the denoiser for the space.");
        }
    }
}
