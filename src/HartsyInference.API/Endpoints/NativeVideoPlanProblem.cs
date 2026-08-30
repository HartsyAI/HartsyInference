using HartsyInference.Engine.Planning;

namespace HartsyInference.API.Endpoints;

/// <summary>Typed HTTP 422 body returned when native video preflight finds an unsafe combination.</summary>
public sealed class NativeVideoPlanProblem
{
    /// <summary>Stable problem type.</summary>
    public string Type { get; init; } = "https://hartsy.ai/problems/video-plan-invalid";

    /// <summary>Human-readable problem title.</summary>
    public string Title { get; init; } = "Video request failed preflight";

    /// <summary>HTTP status code.</summary>
    public int Status { get; init; } = StatusCodes.Status422UnprocessableEntity;

    /// <summary>Machine-readable blocking and warning diagnostics.</summary>
    public required IReadOnlyList<VideoPlanIssue> Issues { get; init; }

    /// <summary>Client-safe resolved values alongside errors. Null only when model/path resolution failed before
    /// a plan could be constructed.</summary>
    public NativeVideoPlanResponse? Plan { get; init; }
}
