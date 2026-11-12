namespace HartsyInference.Engine.Planning;

/// <summary>A stable machine-readable preflight diagnostic of a mesh request, with a human explanation.</summary>
public sealed record MeshPlanIssue
{
    /// <summary>Stable identifier for a host's mapping, for example <c>mesh.field.unsupported</c>.</summary>
    public required string Code { get; init; }

    /// <summary>Whether the issue prevents the run.</summary>
    public required MeshPlanIssueSeverity Severity { get; init; }

    /// <summary>Actionable explanation.</summary>
    public required string Message { get; init; }

    /// <summary>The <see cref="Requests.MeshRequest"/> property the issue concerns, or null for a model-wide problem.</summary>
    public string? Field { get; init; }
}
