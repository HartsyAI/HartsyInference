namespace HartsyInference.API.Endpoints;

/// <summary>Terminal SSE payload emitted when generation fails after response streaming has begun.</summary>
public sealed class NativeSseErrorEvent
{
    /// <summary>Human-readable failure detail.</summary>
    public required string Message { get; init; }
}
