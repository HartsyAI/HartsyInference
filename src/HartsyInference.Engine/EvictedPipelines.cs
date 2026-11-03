using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine;

/// <summary>Pipelines taken out of the cache by a model switch, disposed later under the device gate.</summary>
internal readonly record struct EvictedPipelines(
    IReadOnlyList<KeyValuePair<string, IRecipePipeline>> Images,
    IReadOnlyList<KeyValuePair<string, IVideoRecipePipeline>> Videos)
{
    public int Count => Images.Count + Videos.Count;
}
