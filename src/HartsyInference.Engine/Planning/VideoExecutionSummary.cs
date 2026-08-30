namespace HartsyInference.Engine.Planning;

/// <summary>Auditable effective settings and component formats from one completed video generation.</summary>
public sealed record VideoExecutionSummary
{
    /// <summary>Detected profile identifier.</summary>
    public required string ProfileId { get; init; }

    /// <summary>Conditioning task actually executed.</summary>
    public required VideoTaskFamily Task { get; init; }

    /// <summary>Acceleration path actually executed.</summary>
    public required VideoAccelerationKind Acceleration { get; init; }

    /// <summary>Attention semantics actually executed.</summary>
    public required VideoAttentionKind Attention { get; init; }

    /// <summary>Aligned output width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Aligned output height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Aligned output frame count before optional trimming or boomerang.</summary>
    public required int Frames { get; init; }

    /// <summary>Playback frame rate.</summary>
    public required int Fps { get; init; }

    /// <summary>Concrete generation seed.</summary>
    public required long Seed { get; init; }

    /// <summary>Denoising evaluation count.</summary>
    public required int Steps { get; init; }

    /// <summary>Classifier-free guidance scale.</summary>
    public required float CfgScale { get; init; }

    /// <summary>Effective video flow shift when the profile declares it; null for a legacy family-owned value.</summary>
    public float? FlowShift { get; init; }

    /// <summary>Effective audio flow shift when the profile declares it.</summary>
    public float? AudioFlowShift { get; init; }

    /// <summary>Numerical sampler when the family exposes a named solver.</summary>
    public string? Sampler { get; init; }

    /// <summary>Sigma scheduler when the family exposes a named schedule.</summary>
    public string? Scheduler { get; init; }

    /// <summary>Human-readable PDD/VSA/dense execution path.</summary>
    public required string ExecutionPath { get; init; }

    /// <summary>Actual checkpoint, VAE, adapter, and control formats used by construction.</summary>
    public IReadOnlyDictionary<string, string> ComponentFormats { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);
}
