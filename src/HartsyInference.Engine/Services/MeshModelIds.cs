namespace HartsyInference.Engine.Services;

/// <summary>Catalog ids of the 3D models with a wired pipeline, grouped by the input they take.</summary>
public static class MeshModelIds
{
    /// <summary>Pixal3D: image to mesh, shape-only or textured, with projected DINOv3 features.</summary>
    public const string Pixal3D = "pixal3d";

    /// <summary>The Hunyuan3D-2 shape builds: each id is its own checkpoint folder, and the checkpoint's <c>config.yaml</c> sets the architecture.</summary>
    public static readonly IReadOnlyList<string> Hunyuan3D = ["hunyuan3d", "hunyuan3d-fast", "hunyuan3d-turbo", "hunyuan3d-mini"];

    /// <summary>Image-to-3D models: their request carries <see cref="Requests.MeshRequest.Image"/>.</summary>
    public static readonly IReadOnlyList<string> ImageToMesh = ["triposr", .. Hunyuan3D, "trellis", "trellis2", Pixal3D];

    /// <summary>Scene-decomposition models: their request carries <see cref="Requests.MeshRequest.SceneMesh"/>.</summary>
    public static readonly IReadOnlyList<string> SceneMesh = ["cola3d"];

    /// <summary>Rigging models: their request carries <see cref="Requests.MeshRequest.SceneMesh"/> and the result is a rigged GLB.</summary>
    public static readonly IReadOnlyList<string> RigMesh = ["unirig"];

    /// <summary>Texturing models (Hunyuan3D Paint 2.0 albedo and 2.1 PBR): their request carries both <see cref="Requests.MeshRequest.SceneMesh"/> and
    /// <see cref="Requests.MeshRequest.Image"/> and the result is a textured GLB.</summary>
    public static readonly IReadOnlyList<string> PaintMesh = ["hunyuan3d-paint", "hunyuan3d-paint21"];

    /// <summary>Part-segmentation models (Hunyuan3D-Part P3-SAM): their request carries <see cref="Requests.MeshRequest.SceneMesh"/>, and the result is a GLB with one
    /// coloured node per part plus <see cref="Requests.MeshResult.Parts"/>.</summary>
    public static readonly IReadOnlyList<string> PartMesh = ["hunyuan3d-part"];

    /// <summary>Every id with a wired pipeline; anything else is rejected before a checkpoint loads.</summary>
    public static readonly IReadOnlyList<string> All = [.. ImageToMesh, .. SceneMesh, .. RigMesh, .. PaintMesh, .. PartMesh];
}
