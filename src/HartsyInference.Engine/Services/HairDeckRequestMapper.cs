using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.Engine.Services;

/// <summary>Maps a mesh request onto HairDeck's options, and rejects the HairDeck-only mesh options for every other model.</summary>
internal static class HairDeckRequestMapper
{
    /// <summary>Euler steps of the reference sampler, used when the request names none.</summary>
    public const int DefaultSteps = 50;

    /// <summary>Guidance scale of the reference sampler.</summary>
    public const double DefaultCfg = 2.0;

    /// <summary>The HairDeck options of a request; a negative request seed draws a random one.</summary>
    /// <exception cref="NotSupportedException">The request asks for more than one sample, or for rigged output without chains.</exception>
    public static HairDeckOptions ToOptions(MeshRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        long seed = request.Seed < 0 ? Random.Shared.NextInt64(0, int.MaxValue) : request.Seed;
        return new HairDeckOptions
        {
            Seed = seed,
            Steps = request.Steps > 0 ? request.Steps : DefaultSteps,
            Cfg = DefaultCfg,
            Samples = request.Samples,
            Colors = request.Colors ?? true,
            Chains = request.Chains ?? true,
            Rigged = request.Rigged,
        }.Validate();
    }

    /// <summary>Fails when a HairDeck-only option is set for another model, so the request is never silently changed.</summary>
    /// <exception cref="NotSupportedException">A sample count other than 1, rigged output, or a colour or chain choice on a model that has none.</exception>
    public static void RejectForOtherModels(MeshRequest request, string modelId)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Samples != 1)
        {
            throw new NotSupportedException($"'{modelId}' generates one sample per request; {request.Samples} samples were requested.");
        }

        if (request.Rigged)
        {
            throw new NotSupportedException($"'{modelId}' has no rigged output; rigged output is HairDeck-only.");
        }

        if (request.Colors is not null || request.Chains is not null)
        {
            throw new NotSupportedException($"'{modelId}' takes no colour or chain option; those are HairDeck-only.");
        }
    }
}
