namespace HartsyInference.Engine.Services;

/// <summary>Catalog ids of the 3D models with a wired pipeline, grouped by the input they take.</summary>
public static class MeshModelIds
{
    /// <summary>Image-to-3D models: their request carries <see cref="Requests.MeshRequest.Image"/>.</summary>
    public static readonly IReadOnlyList<string> ImageToMesh = ["triposr", "hunyuan3d", "trellis"];

    /// <summary>Scene-decomposition models: their request carries <see cref="Requests.MeshRequest.SceneMesh"/>.</summary>
    public static readonly IReadOnlyList<string> SceneMesh = ["cola3d"];

    /// <summary>Rigging models: their request carries <see cref="Requests.MeshRequest.SceneMesh"/> and the result is a rigged GLB.</summary>
    public static readonly IReadOnlyList<string> RigMesh = ["unirig"];

    /// <summary>Every id with a wired pipeline; anything else is rejected before a checkpoint loads.</summary>
    public static readonly IReadOnlyList<string> All = [.. ImageToMesh, .. SceneMesh, .. RigMesh];
}
