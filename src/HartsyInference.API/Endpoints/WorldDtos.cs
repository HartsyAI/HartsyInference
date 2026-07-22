using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>POST /v1/native/world/sessions</c>.</summary>
public sealed class NativeWorldRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required WorldRequest Request { get; set; }
}

/// <summary>Request body for <c>POST /v1/native/world/sessions/{id}/action</c>.</summary>
public sealed class WorldActionRequest
{
    /// <summary>Host-defined action/control token fed into the world's next step.</summary>
    public required string Action { get; set; }
}
