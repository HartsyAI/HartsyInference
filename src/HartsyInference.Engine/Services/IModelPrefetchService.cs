using HartsyInference.Audio.Cache;
using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine.Services;

/// <summary>Downloads a model's weights without loading them, so an installer can fetch a checkpoint ahead of
/// the first generation and report real progress and real failures.</summary>
public interface IModelPrefetchService
{
    /// <summary>Fetches everything <paramref name="spec"/>'s model needs, skipping whatever is already cached.
    ///
    /// <para>The primary artifact is fetched last, so an interrupted or failed prefetch cannot leave a
    /// checkpoint that looks complete on disk but is missing a companion file.</para></summary>
    Task<ModelPrefetchResult> PrefetchAsync(ModelSpec spec, IProgress<AudioFetchProgress>? progress = null,
        CancellationToken cancel = default);
}
