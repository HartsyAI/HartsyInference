namespace HartsyInference.Engine.Planning;

/// <summary>Local hash-bound profile data for a structurally valid H3 community checkpoint absent from the built-in manifest.</summary>
internal sealed record VideoProfileSidecar
{
    /// <summary>Full-file SHA-256 of the exact transformer artifact this sidecar describes.</summary>
    public required string Sha256 { get; init; }

    /// <summary>Stable operator-chosen profile id.</summary>
    public required string ProfileId { get; init; }

    /// <summary>Operator-facing variant name.</summary>
    public string? DisplayName { get; init; }

    /// <summary>Conditioning task certified by the sidecar author.</summary>
    public required VideoTaskFamily Task { get; init; }

    /// <summary>Acceleration semantics baked into the checkpoint.</summary>
    public VideoAccelerationKind Acceleration { get; init; }

    /// <summary>Attention semantics required by the checkpoint.</summary>
    public VideoAttentionKind Attention { get; init; }

    /// <summary>Validated denoising evaluation count.</summary>
    public required int Steps { get; init; }

    /// <summary>Validated guidance scale.</summary>
    public float CfgScale { get; init; } = 1f;

    /// <summary>Validated video flow shift.</summary>
    public float FlowShift { get; init; } = 12f;

    /// <summary>Validated audio flow shift.</summary>
    public float AudioFlowShift { get; init; } = 3f;

    /// <summary>Validated sampler.</summary>
    public string Sampler { get; init; } = "euler";

    /// <summary>Validated scheduler.</summary>
    public string Scheduler { get; init; } = "normal";

    /// <summary>Optional locked target width.</summary>
    public int? Width { get; init; }

    /// <summary>Optional locked target height.</summary>
    public int? Height { get; init; }

    /// <summary>Reference-media sizing policy.</summary>
    public VideoReferenceSizing ReferenceSizing { get; init; }

    /// <summary>Artifact provenance page.</summary>
    public string? ProvenanceUrl { get; init; }

    /// <summary>Artifact license page.</summary>
    public string? LicenseUrl { get; init; }
}
