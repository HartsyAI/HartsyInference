namespace HartsyInference.Engine.Services;

/// <summary>Outcome of a <see cref="IModelPrefetchService.PrefetchAsync"/> call.
///
/// <para><see cref="Supported"/> is false for families whose weights are not wired for prefetch yet. That is
/// reported rather than thrown so an installer can say what actually happened — the alternative that existed
/// before this API was to claim success and let the download happen silently on first generation.</para></summary>
/// <param name="Supported">Whether this model family can be prefetched by this build.</param>
/// <param name="Message">Human-readable outcome, suitable for an install log.</param>
/// <param name="Files">Absolute paths fetched or already cached; empty when unsupported.</param>
/// <param name="PrimaryPath">The model's entrypoint artifact — the file whose presence means "this model is installed", and the one a caller attaches identity metadata to. Null when unsupported. Stated explicitly because the position of the primary within <paramref name="Files"/> is an ordering detail, not a contract.</param>
public sealed record ModelPrefetchResult(bool Supported, string Message, IReadOnlyList<string> Files, string? PrimaryPath = null)
{
    /// <summary>Result for a family this build cannot prefetch.</summary>
    public static ModelPrefetchResult Unsupported(string id, string reason)
        => new(false, $"'{id}' cannot be pre-downloaded by this build: {reason}. Its weights are fetched on first use.", []);
}
