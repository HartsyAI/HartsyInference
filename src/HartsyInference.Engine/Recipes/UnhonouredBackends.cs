using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Recipes;

/// <summary>Checks the placement backends a recipe's pipeline does not honour, against <see cref="RecipeContext.Backend"/>. The
/// engine sets every placement backend on the context for every recipe, so a pipeline that runs a stage on the primary backend
/// would otherwise ignore a configured device without saying so.
/// <list type="bullet">
/// <item>The text encoder and the VAE are refused with a warning and run on the primary backend. A host that places them on the CPU
/// or a second card for every model must still get FLUX.2 and FLUX.3 to run.</item>
/// <item>CFG-parallel, DiT sharding and context parallelism change the transformer's placement, so they are refused with an error.</item>
/// </list></summary>
public static class UnhonouredBackends
{
    /// <summary>Throws <see cref="NotSupportedException"/> for a set transformer placement, and logs a warning for a set text-encoder or
    /// VAE placement. A null or primary-equal backend is the default and passes.</summary>
    /// <param name="context">The recipe context the engine built for the request.</param>
    /// <param name="family">The model family named in the messages, for example <c>FLUX.2</c>.</param>
    /// <remarks>After this returns, <see cref="RecipeContext.TransformerBackends"/> is the primary backend alone: its other members are the
    /// three refused placements, and they are the only ones that add a transformer backend.</remarks>
    public static void Refuse(RecipeContext context, string family)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentException.ThrowIfNullOrWhiteSpace(family);

        List<string> transformer = [];
        AddIfSeparate(transformer, context, context.CfgParallelBackend, "CfgParallelBackend");
        AddIfSeparate(transformer, context, context.DitShardBackend, "DitShardBackend");
        if (context.DitShardBackends is not null && context.DitShardBackends.Any(b => !ReferenceEquals(b, context.Backend)))
        {
            transformer.Add("DitShardBackends");
        }
        if (context.CpBackends is not null && context.CpBackends.Any(b => !ReferenceEquals(b, context.Backend)))
        {
            transformer.Add("CpBackends");
        }
        if (transformer.Count > 0)
        {
            throw new NotSupportedException(
                $"{family} runs the transformer on the primary backend only, so {string.Join(", ", transformer)} cannot be honoured. " +
                "Remove --cfg-parallel-gpu, --dit-shard-gpu and --cp-gpu for this model.");
        }

        List<string> primary = [];
        AddIfSeparate(primary, context, context.TextEncoderBackend, "TextEncoderBackend");
        AddIfSeparate(primary, context, context.VaeBackend, "VaeBackend");
        if (primary.Count > 0)
        {
            Logs.Warning($"{family} runs the {string.Join(" and the ", primary)} on the primary backend; the placement set by --te-gpu " +
                "and --vae-gpu is ignored for this model.");
        }
    }

    private static void AddIfSeparate(List<string> set, RecipeContext context, IBackend? backend, string name)
    {
        if (backend is not null && !ReferenceEquals(backend, context.Backend))
        {
            set.Add(name);
        }
    }
}
