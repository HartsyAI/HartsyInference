namespace HartsyInference.API.Endpoints;

/// <summary>Body of <c>POST /v1/native/mesh/prefetch</c>: names the 3D model whose missing files to fetch.</summary>
public sealed class NativeMeshPrefetchRequest
{
    /// <summary>Catalog id of the 3D model.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint folder, as on the generation routes; usually left null.</summary>
    public string? ModelPath { get; set; }
}
