namespace HartsyInference.ThreeD.Geometry;

/// <summary>An unstructured point set in object space (positions, optional per-point colors/normals).
/// Produced by surface sampling (<see cref="Ops.SurfaceSampler"/>) for VecSet-style encoders and as an
/// intermediate for splat/point pipelines. Flat interleaved arrays, x,y,z per point.</summary>
public sealed class PointCloud
{
    /// <summary>Point positions, length <c>3 * Count</c>.</summary>
    public required float[] Positions { get; init; }

    /// <summary>Per-point RGB in [0,1], length <c>3 * Count</c>, or null.</summary>
    public float[]? Colors { get; set; }

    /// <summary>Per-point normals, length <c>3 * Count</c>, or null.</summary>
    public float[]? Normals { get; set; }

    /// <summary>Number of points.</summary>
    public int Count => Positions.Length / 3;
}
