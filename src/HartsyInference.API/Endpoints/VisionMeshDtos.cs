using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/vision</c>. <see cref="VisionRequest.Image"/> is raw RGB24
/// (<see cref="ImageData"/>), matching the native contract exactly — no PNG/JPEG decode convenience layer, same
/// as every other native route's pass-through philosophy.</summary>
public sealed class NativeVisionRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required VisionRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/mesh</c>.</summary>
public sealed class NativeMeshRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required MeshRequest Request { get; set; }
}
