namespace HartsyInference.API.Endpoints;

/// <summary>Body of <c>POST /v1/native/mesh/free</c>.</summary>
public sealed class NativeMeshFreeRequest
{
    /// <summary>The <c>name</c> of a resident 3D model from <c>GET /v1/native/mesh/loaded</c>; null or absent frees every one.</summary>
    public string? Name { get; set; }
}
