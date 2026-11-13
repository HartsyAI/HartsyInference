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

    /// <summary>Remove edge-connected components with fewer faces than this fraction (between 0 and 1) of the largest component's face count, truncated as MeshLab truncates it; 0 disables. This is
    /// pymeshlab's <c>compute_selection_by_small_disconnected_components_per_face(nbfaceratio)</c>, which <c>hy3dshape/postprocessors.py</c> (<c>remove_floater</c>) calls with 0.005; the engine applies it
    /// to Hunyuan3D-Omni so far (the 2.x shape models ship the same post-processor, their engine path does not use the rule yet). It runs before <see cref="SmallComponentAreaRatio"/>.</summary>
    public double SmallComponentFaceRatio { get; init; }

    /// <summary>Fill boundary loops up to this perimeter; 0 disables.</summary>
    public double HoleFillMaxPerimeter { get; init; } = 3e-2;

    /// <summary>After the perimeter fill, close the boundary loops that <see cref="HoleFiller"/> can triangulate with no new edge longer than this; 0 disables.
    /// A thin slit is closed whatever its length, a wide opening such as a cut base or a mouth is not. With <see cref="UnifyOrientation"/> the winding is unified before the fill as well as after it, because the patches take their winding from the faces around the hole.</summary>
    public double HoleFillMaxWidth { get; init; }

    /// <summary>Make neighboring faces agree on winding.</summary>
    public bool UnifyOrientation { get; init; } = true;

    /// <summary>After unifying, flip closed components whose signed volume is negative so normals point outward.</summary>
    public bool OrientOutward { get; init; } = true;
}
