namespace HartsyInference.Engine.Planning;

/// <summary>A stable machine-readable preflight diagnostic with a human explanation.</summary>
public sealed record VideoPlanIssue
{
    /// <summary>Stable identifier suitable for HTTP problem details and UI mapping.</summary>
    public required string Code { get; init; }

    /// <summary>Whether this issue prevents generation.</summary>
    public required VideoPlanIssueSeverity Severity { get; init; }

    /// <summary>Actionable explanation for the operator.</summary>
    public required string Message { get; init; }

    /// <summary>Request field associated with the issue, or null for an artifact-wide problem.</summary>
    public string? Field { get; init; }
}
