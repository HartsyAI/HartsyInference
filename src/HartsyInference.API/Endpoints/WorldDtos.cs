using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>POST /v1/native/world/sessions</c>.</summary>
public sealed class NativeWorldRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native world-session seed request, unmodified.</summary>
    public required WorldRequest Request { get; set; }
}

/// <summary>Request body for <c>POST /v1/native/world/sessions/{id}/action</c>.</summary>
public sealed class WorldActionRequest
{
    /// <summary>Host-defined action/control token fed into the world's next step.</summary>
    public required string Action { get; set; }
}
