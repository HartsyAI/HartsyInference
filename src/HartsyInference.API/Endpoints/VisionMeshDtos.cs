using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/vision</c>. <see cref="VisionRequest.Image"/> is raw RGB24
/// (<see cref="ImageData"/>), matching the native contract exactly — no PNG/JPEG decode convenience layer, same
/// as every other native route's pass-through philosophy.</summary>
public sealed class NativeVisionRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a
    /// relative one resolves against the server process's working directory, not anything an HTTP client can
    /// know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native vision request, unmodified.</summary>
    public required VisionRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/mesh</c>.</summary>
public sealed class NativeMeshRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a
    /// relative one resolves against the server process's working directory, not anything an HTTP client can
    /// know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native mesh request, unmodified.</summary>
    public required MeshRequest Request { get; set; }
}
