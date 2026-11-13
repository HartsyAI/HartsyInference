namespace HartsyInference.Engine.Services;

/// <summary>Catalog ids of the 3D models with a wired pipeline, grouped by the input they take.</summary>
public static class MeshModelIds
{
    /// <summary>HairDeck: image to hair cards; it has its own request options and is not part of <see cref="ImageToMesh"/>.</summary>
    public const string HairDeck = "hairdeck";

    /// <summary>TRELLIS.2: image to mesh, shape-only or textured.</summary>
    public const string Trellis2 = "trellis2";

    /// <summary>Pixal3D: image to mesh, shape-only or textured, with projected DINOv3 features.</summary>
    public const string Pixal3D = "pixal3d";

    /// <summary>SAM 3D Objects: image plus object mask to Gaussian splats (PLY); the mask is the request image's alpha plane and MoGe estimates the pointmap.</summary>
    public const string Sam3dObjects = "sam3d-objects";

    /// <summary>The Hunyuan3D-2 shape builds: each id is its own checkpoint folder, and the checkpoint's <c>config.yaml</c> sets the architecture.</summary>
    public static readonly IReadOnlyList<string> Hunyuan3D = ["hunyuan3d", "hunyuan3d-fast", "hunyuan3d-turbo", "hunyuan3d-mini"];

    /// <summary>Image-to-3D models: their request carries <see cref="Requests.MeshRequest.Image"/>.</summary>
    public static readonly IReadOnlyList<string> ImageToMesh = ["triposr", .. Hunyuan3D, "trellis", "trellis2", Pixal3D, Sam3dObjects];

    /// <summary>Scene-decomposition models: their request carries <see cref="Requests.MeshRequest.SceneMesh"/>.</summary>
    public static readonly IReadOnlyList<string> SceneMesh = ["cola3d"];

    /// <summary>Rigging models: their request carries <see cref="Requests.MeshRequest.SceneMesh"/> and the result is a rigged GLB.</summary>
    public static readonly IReadOnlyList<string> RigMesh = ["unirig"];

    /// <summary>Texturing models (Hunyuan3D Paint 2.0 albedo and 2.1 PBR): their request carries both <see cref="Requests.MeshRequest.SceneMesh"/> and
    /// <see cref="Requests.MeshRequest.Image"/> and the result is a textured GLB.</summary>
    public static readonly IReadOnlyList<string> PaintMesh = ["hunyuan3d-paint", "hunyuan3d-paint21"];

    /// <summary>Part-segmentation models (Hunyuan3D-Part P3-SAM): their request carries <see cref="Requests.MeshRequest.SceneMesh"/>, and the result is a GLB with one
    /// coloured node per part plus <see cref="Requests.MeshResult.Parts"/>.</summary>
    public static readonly IReadOnlyList<string> PartMesh = [P3Sam];

    /// <summary>The id of Hunyuan3D-Part P3-SAM.</summary>
    public const string P3Sam = "hunyuan3d-part";

    /// <summary>The id of Hunyuan3D-Part X-Part.</summary>
    public const string XPart = "hunyuan3d-xpart";

    /// <summary>Part-generation models (Hunyuan3D-Part X-Part): the request carries <see cref="Requests.MeshRequest.SceneMesh"/>, P3-SAM splits it into parts and every part is generated again as a
    /// complete shape; the result is a GLB with one coloured node per part plus <see cref="Requests.MeshResult.Parts"/>.</summary>
    public static readonly IReadOnlyList<string> XPartMesh = [XPart];

    /// <summary>The id of Hunyuan3D-Omni.</summary>
    public const string Omni = "hunyuan3d-omni";

    /// <summary>Controlled shape generation (Hunyuan3D-Omni): the request carries <see cref="Requests.MeshRequest.Image"/> and <see cref="Requests.MeshRequest.Control"/>.</summary>
    public static readonly IReadOnlyList<string> OmniMesh = [Omni];

    /// <summary>Every id with a wired pipeline; anything else is rejected before a checkpoint loads.</summary>
    public static readonly IReadOnlyList<string> All = [.. ImageToMesh, .. SceneMesh, .. RigMesh, .. PaintMesh, .. PartMesh, .. XPartMesh, .. OmniMesh];

    /// <summary><see cref="All"/> and <see cref="HairDeck"/>: every id a host can describe and plan.</summary>
    public static readonly IReadOnlyList<string> Described = [.. All, HairDeck];
}
