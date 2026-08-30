using HartsyInference.Engine.Planning;

namespace HartsyInference.API.Endpoints;

/// <summary>Payload of the terminal <c>complete</c> event from <c>/v1/native/video/stream</c>.</summary>
public sealed class NativeVideoCompleteEvent
{
    /// <summary>Number of emitted frames.</summary>
    public required int Frames { get; init; }

    /// <summary>Persisted output directory, or null when saving was disabled or failed.</summary>
    public string? SavedPath { get; init; }

    /// <summary>Auditable profile, sampling, geometry, and component formats actually executed.</summary>
    public VideoExecutionSummary? Execution { get; init; }
}
