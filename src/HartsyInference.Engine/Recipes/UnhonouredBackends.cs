using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Recipes;

/// <summary>Refuses the placement backends a recipe's pipeline does not honour. A pipeline that runs every stage on
/// <see cref="RecipeContext.Backend"/> would otherwise ignore a text-encoder, VAE, CFG-parallel, DiT-shard or
/// context-parallel backend the placement set, and the run would look configured while it was not.</summary>
public static class UnhonouredBackends
{
    /// <summary>Throws <see cref="NotSupportedException"/> naming each placement backend in <paramref name="context"/> that is set
    /// and is not the primary backend. A null or primary-equal backend is the default and passes.</summary>
    /// <param name="context">The recipe context the engine built for the request.</param>
    /// <param name="family">The model family named in the message, for example <c>FLUX.2</c>.</param>
    public static void Refuse(RecipeContext context, string family)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentException.ThrowIfNullOrWhiteSpace(family);

        List<string> set = [];
        AddIfSeparate(set, context, context.TextEncoderBackend, "TextEncoderBackend");
        AddIfSeparate(set, context, context.VaeBackend, "VaeBackend");
        AddIfSeparate(set, context, context.CfgParallelBackend, "CfgParallelBackend");
        AddIfSeparate(set, context, context.DitShardBackend, "DitShardBackend");
        if (context.DitShardBackends is not null && context.DitShardBackends.Any(b => !ReferenceEquals(b, context.Backend)))
        {
            set.Add("DitShardBackends");
        }
        if (context.CpBackends is not null && context.CpBackends.Any(b => !ReferenceEquals(b, context.Backend)))
        {
            set.Add("CpBackends");
        }

        if (set.Count > 0)
        {
            throw new NotSupportedException(
                $"{family} runs every stage on the primary backend and does not honour {string.Join(", ", set)}. " +
                "Clear those placement devices for this model; the run would otherwise look configured while it ran on one device.");
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
