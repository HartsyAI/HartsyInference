using HartsyInference.Engine.Planning;

namespace HartsyInference.API.Endpoints;

/// <summary>Typed HTTP 422 body for a describe request whose model cannot be described.</summary>
public sealed class NativeRecipeProblem
{
    /// <summary>Stable problem type.</summary>
    public string Type { get; init; } = "https://hartsy.ai/problems/recipe-invalid";

    /// <summary>Human-readable problem title.</summary>
    public string Title { get; init; } = "Model cannot be described";

    /// <summary>HTTP status code.</summary>
    public int Status { get; init; } = StatusCodes.Status422UnprocessableEntity;

    /// <summary>Machine-readable diagnostics, with local paths removed.</summary>
    public required IReadOnlyList<VideoPlanIssue> Issues { get; init; }
}
