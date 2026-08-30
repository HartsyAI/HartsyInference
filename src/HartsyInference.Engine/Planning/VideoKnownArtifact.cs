namespace HartsyInference.Engine.Planning;

/// <summary>One full-file SHA-256 entry with enough semantics to build or validate a profile composition.</summary>
internal sealed record VideoKnownArtifact
{
    /// <summary>Lowercase full-file SHA-256.</summary>
    public required string Sha256 { get; init; }

    /// <summary>Stable artifact/profile identifier.</summary>
    public required string Id { get; init; }

    /// <summary>Operator-facing artifact name.</summary>
    public required string DisplayName { get; init; }

    /// <summary>Composition role.</summary>
    public required VideoProfileArtifactRole Role { get; init; }

    /// <summary>Conditioning task declared for the artifact.</summary>
    public required VideoTaskFamily Task { get; init; }

    /// <summary>Acceleration semantics introduced by the artifact.</summary>
    public required VideoAccelerationKind Acceleration { get; init; }

    /// <summary>Attention semantics introduced by the artifact.</summary>
    public VideoAttentionKind Attention { get; init; } = VideoAttentionKind.Dense;

    /// <summary>Declared evaluation count, or null when the base recipe supplies it.</summary>
    public int? Steps { get; init; }

    /// <summary>Declared video flow shift, or null when the base recipe supplies it.</summary>
    public float? FlowShift { get; init; }

    /// <summary>Declared audio flow shift, or null when the base recipe supplies it.</summary>
    public float? AudioFlowShift { get; init; }

    /// <summary>Declared target width, or null when request geometry remains free.</summary>
    public int? Width { get; init; }

    /// <summary>Declared target height, or null when request geometry remains free.</summary>
    public int? Height { get; init; }

    /// <summary>Reference-media sizing policy associated with the artifact.</summary>
    public VideoReferenceSizing ReferenceSizing { get; init; } = VideoReferenceSizing.Native;

    /// <summary>Reason an artifact may not execute, for <see cref="VideoProfileArtifactRole.Rejected"/> entries.</summary>
    public string? RejectionReason { get; init; }

    /// <summary>Artifact provenance page.</summary>
    public string? ProvenanceUrl { get; init; }
}
