using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/video/stream</c>.</summary>
public sealed class NativeVideoRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required VideoRequest Request { get; set; }
}
