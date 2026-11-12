using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Body of <c>POST /v1/native/mesh/thumbnail</c>.</summary>
public sealed class NativeMeshThumbnailRequest
{
    /// <summary>The model file, base64.</summary>
    public string? Data { get; set; }

    /// <summary>The file's format: <c>glb</c>, <c>obj</c> or <c>ply</c> (a splat or point cloud).</summary>
    public string? Format { get; set; }

    /// <summary>Size, view and background; absent takes the defaults.</summary>
    public MeshThumbnailRequest? Options { get; set; }
}
