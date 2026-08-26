namespace HartsyInference.Engine.Services;

/// <summary>Outcome of a <see cref="IModelPrefetchService.PrefetchAsync"/> call.
///
/// <para><see cref="Supported"/> is false for families whose weights are not wired for prefetch yet. That is
/// reported rather than thrown so an installer can say what actually happened — the alternative that existed
/// before this API was to claim success and let the download happen silently on first generation.</para></summary>
/// <param name="Supported">Whether this model family can be prefetched by this build.</param>
/// <param name="Message">Human-readable outcome, suitable for an install log.</param>
/// <param name="Files">Absolute paths fetched or already cached; empty when unsupported.</param>
public sealed record ModelPrefetchResult(bool Supported, string Message, IReadOnlyList<string> Files)
{
    /// <summary>Result for a family this build cannot prefetch.</summary>
    public static ModelPrefetchResult Unsupported(string id, string reason)
        => new(false, $"'{id}' cannot be pre-downloaded by this build: {reason}. Its weights are fetched on first use.", []);
}
