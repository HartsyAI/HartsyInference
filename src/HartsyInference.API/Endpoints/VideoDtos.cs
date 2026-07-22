using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/video/stream</c>.</summary>
public sealed class NativeVideoRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a
    /// relative one resolves against the server process's working directory, not anything an HTTP client can
    /// know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native video request, unmodified.</summary>
    public required VideoRequest Request { get; set; }
}
