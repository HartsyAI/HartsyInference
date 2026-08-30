using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine.Planning;

/// <summary>A hash- and header-resolved video checkpoint contract, independent of its filename.</summary>
public sealed record VideoModelProfile
{
    /// <summary>Stable identifier accepted by <c>ModelSpec.ProfileId</c>.</summary>
    public required string Id { get; init; }

    /// <summary>Operator-facing checkpoint or composition name.</summary>
    public required string DisplayName { get; init; }

    /// <summary>Engine recipe family that constructs the model.</summary>
    public required string FamilyId { get; init; }

    /// <summary>Conditioning family declared by this hash-bound artifact or composition.</summary>
    public required VideoTaskFamily Task { get; init; }

    /// <summary>Checkpoint-bound acceleration semantics.</summary>
    public required VideoAccelerationKind Acceleration { get; init; }

    /// <summary>Attention implementation the profile requires.</summary>
    public required VideoAttentionKind Attention { get; init; }

    /// <summary>Hash-bound sampling defaults and field locks.</summary>
    public required VideoDefaults Defaults { get; init; }

    /// <summary>Conditioning inputs the profile can consume without silently discarding them.</summary>
    public required VideoFeatures Features { get; init; }

    /// <summary>Full-file SHA-256 when the profile is bound to a known artifact.</summary>
    public string? ArtifactSha256 { get; init; }

    /// <summary>Weight format inferred from tensor descriptors and quantization companions.</summary>
    public string? Quantization { get; init; }

    /// <summary>Whether the profile came from Hartsy's built-in manifest.</summary>
    public bool IsBuiltIn { get; init; }

    /// <summary>Whether the profile came from a hash-bound local sidecar.</summary>
    public bool IsSidecar { get; init; }

    /// <summary>Checkpoint provenance page shown to operators.</summary>
    public string? ProvenanceUrl { get; init; }

    /// <summary>Artifact or base-model license page shown to operators.</summary>
    public string? LicenseUrl { get; init; }

    /// <summary>Safetensors metadata copied from the header before its mmap is released.</summary>
    public IReadOnlyDictionary<string, string> CheckpointMetadata { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);
}
