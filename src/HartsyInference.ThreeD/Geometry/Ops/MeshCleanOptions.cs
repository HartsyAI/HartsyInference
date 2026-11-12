namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Switches and thresholds for <see cref="MeshCleaner.Clean"/>. Lengths are in mesh units (object space), not relative to the bounding box.
/// Defaults follow the upstream TRELLIS.2 cleanup between decimation passes.</summary>
public sealed record MeshCleanOptions
{
    /// <summary>Merge vertices at identical positions first.</summary>
    public bool WeldVertices { get; init; } = true;

    /// <summary>Weld distance in mesh units; 0 merges only bit-identical positions.</summary>
    public double WeldTolerance { get; init; }

    /// <summary>Drop faces that repeat a vertex or have zero area.</summary>
    public bool RemoveDegenerateFaces { get; init; } = true;

    /// <summary>Drop faces that reuse another face's three vertices (either winding).</summary>
    public bool RemoveDuplicateFaces { get; init; } = true;

    /// <summary>Split vertices so no edge is shared by more than two faces (upstream's <c>repair_non_manifold_edges</c>); no face is removed.</summary>
    public bool RepairNonManifoldEdges { get; init; } = true;

    /// <summary>Remove vertex-connected components whose area is below this fraction of the total area; 0 disables.</summary>
    public double SmallComponentAreaRatio { get; init; } = 1e-5;

    /// <summary>Fill boundary loops up to this perimeter; 0 disables.</summary>
    public double HoleFillMaxPerimeter { get; init; } = 3e-2;

    /// <summary>Make neighboring faces agree on winding.</summary>
    public bool UnifyOrientation { get; init; } = true;

    /// <summary>After unifying, flip closed components whose signed volume is negative so normals point outward.</summary>
    public bool OrientOutward { get; init; } = true;
}
