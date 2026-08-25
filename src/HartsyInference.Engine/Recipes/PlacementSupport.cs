using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Recipes;

/// <summary>Notes a placement that IS in effect, for the cases whose whole symptom is that things quietly work.</summary>
/// <remarks>The counterpart — reporting a setting that is configured and CANNOT take effect — moved to
/// <see cref="MemorySupportReport"/>, which the engine drives for every recipe from a declared
/// <see cref="MemoryCapabilities"/>. That inversion is the point: the per-recipe warning this class used to own
/// reached 2 of 39 recipes, so the rest gave "ignored" and "working" the same silence.</remarks>
public static class PlacementSupport
{
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
