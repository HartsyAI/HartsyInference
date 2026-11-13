namespace HartsyInference.Engine.Requests;

/// <summary>The control signals Hunyuan3D-Omni takes beside its image.</summary>
public enum MeshControlKind
{
    /// <summary>A point cloud (or a mesh's vertices) the shape should follow: one token per point.</summary>
    Point,

    /// <summary>A voxel set: a triangle mesh whose surface is sampled and quantised to 16 voxels per axis.</summary>
    Voxel,

    /// <summary>A bounding box: the three extents (length, height, width), each in (0, 1].</summary>
    BoundingBox,

    /// <summary>A skeleton: bones as <c>x1 y1 z1 x2 y2 z2</c>, six values each, in the model's own frame.</summary>
    Pose,
}

/// <summary>One control signal for Hunyuan3D-Omni: what it is and its data. <see cref="MeshControlKind.Point"/> and <see cref="MeshControlKind.Voxel"/> carry <see cref="Geometry"/>; <see cref="MeshControlKind.BoundingBox"/> and
/// <see cref="MeshControlKind.Pose"/> carry <see cref="Values"/>.</summary>
public sealed record MeshControl
{
    /// <summary>The kind of signal.</summary>
    public required MeshControlKind Kind { get; init; }

    /// <summary>Point: <c>obj</c>, <c>glb</c> or a <c>ply</c> point cloud, whose vertices are the points (normalised into the cube as upstream does). Voxel: an <c>obj</c> or <c>glb</c> with faces. Null for the other kinds.</summary>
    public SceneMeshData? Geometry { get; init; }

    /// <summary>BoundingBox: length, height and width, each in (0, 1]. Pose: the bones, six values each. Null for the other kinds.</summary>
    public IReadOnlyList<float>? Values { get; init; }
}
