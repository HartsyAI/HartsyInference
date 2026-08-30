namespace HartsyInference.API.Endpoints;

/// <summary>Payload of one <c>frame</c> event from <c>/v1/native/video/stream</c>.</summary>
public sealed class NativeVideoFrameEvent
{
    /// <summary>Zero-based frame index.</summary>
    public required int Index { get; init; }

    /// <summary>Base64-encoded PNG frame.</summary>
    public required string Png { get; init; }
}
