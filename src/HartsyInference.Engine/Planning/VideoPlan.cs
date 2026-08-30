using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Planning;

/// <summary>One immutable preflight result that drives video validation, cache identity, construction, and execution.</summary>
public sealed record VideoPlan
{
    /// <summary>The in-memory request instance this plan validated. It is deliberately internal and is not
    /// serialized: executing a deserialized or request-swapped plan would bypass media/control validation.</summary>
    internal VideoRequest? SourceRequest { get; init; }

    /// <summary>Deep-frozen request graph that planning actually validated and execution exclusively consumes.
    /// Kept internal so callers cannot mutate the copied media buffers through the public plan DTO.</summary>
    internal VideoRequest? ExecutionRequest { get; init; }

    /// <summary>Structural SHA-256 of the source request at planning time. Execution re-computes it from the
    /// original instance before using <see cref="ExecutionRequest"/>, rejecting nested list/media mutation.</summary>
    internal string? SourceRequestFingerprint { get; init; }

    /// <summary>Frozen plan state used by execution even when a caller clones the public DTO.</summary>
    internal VideoPlan? ExecutionPlan { get; init; }

    /// <summary>Canonical file identities captured after header and hash resolution.</summary>
    internal IReadOnlyDictionary<string, VideoArtifactFileStamp> ArtifactFileStamps { get; init; } =
        new Dictionary<string, VideoArtifactFileStamp>(StringComparer.Ordinal);

    /// <summary>The model request this plan resolved.</summary>
    public required ModelSpec Model { get; init; }

    /// <summary>Detected checkpoint and adapter profile.</summary>
    public required VideoModelProfile Profile { get; init; }

    /// <summary>Resolved values passed to the recipe pipeline.</summary>
    public required VideoEffectiveSettings EffectiveSettings { get; init; }

    /// <summary>All preflight diagnostics, including non-blocking warnings.</summary>
    public required IReadOnlyList<VideoPlanIssue> Issues { get; init; }

    /// <summary>Stable construction-affecting identity appended to the pipeline cache key.</summary>
    public required string CacheIdentity { get; init; }

    /// <summary>Resolved local component paths keyed by role.</summary>
    public IReadOnlyDictionary<string, string> ComponentPaths { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Actual formats inferred from each resolved component header.</summary>
    public IReadOnlyDictionary<string, string> ComponentFormats { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Full-file hashes already computed for manifest and sidecar validation.</summary>
    public IReadOnlyDictionary<string, string> ArtifactHashes { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Unmodified safetensors metadata keyed by artifact role, including PDD and conversion provenance.</summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> ArtifactMetadata { get; init; } =
        new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal);

    /// <summary>Whether preflight found no blocking issue.</summary>
    public bool IsValid => !Issues.Any(issue => issue.Severity == VideoPlanIssueSeverity.Error);

    /// <summary>Throws a typed exception carrying this plan when execution is unsafe.</summary>
    public void ThrowIfInvalid()
    {
        if (!IsValid)
        {
            throw new VideoPlanningException(this);
        }
    }
}
