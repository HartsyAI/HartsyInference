namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>The last step of a shape-only mesh from a dual-grid decoder (TRELLIS.2, Pixal3D): the repair the textured path gets from
/// <see cref="PbrMeshPostProcessor"/>, without the decimation and the bake. The raw mesh has non-manifold edges and loose specks; the repair
/// welds, drops degenerate and duplicate faces, splits non-manifold edges, removes tiny components, closes small holes and unifies the winding
/// (<see cref="MeshCleanOptions"/> defaults). Boundary edges along larger openings and thin slits remain unless an opening width is given, which also
/// closes every loop <see cref="HoleFiller"/> can zip shut within that width; upstream closes them with a dual-contouring remesh this engine does not have.</summary>
public static class ShapeMeshFinisher
{
    /// <summary>The opening width, as a share of the mesh's bounding-box diagonal, that <see cref="Finish"/> callers pass to close slits and gaps while leaving a cut base or a mouth open.</summary>
    public const double DefaultOpeningWidthRatio = 0.1;

    /// <summary>Returns the mesh to export: repaired with <see cref="MeshCleaner.Clean"/> when <paramref name="clean"/> is set, otherwise
    /// the winding unified only, then smooth vertex normals. The input is not modified when it is repaired. <paramref name="smallComponentAreaRatio"/> is
    /// <see cref="MeshCleanOptions.SmallComponentAreaRatio"/>: a model whose extraction leaves larger floating pieces raises it. <paramref name="smallComponentFaceRatio"/> is
    /// <see cref="MeshCleanOptions.SmallComponentFaceRatio"/>, the rule of a model whose upstream removes floaters by face count (the Hunyuan3D shape models). A positive
    /// <paramref name="openingWidthRatio"/> (<see cref="DefaultOpeningWidthRatio"/>) also closes the boundary loops that triangulate with no new edge longer than that
    /// share of the bounding-box diagonal (<see cref="MeshCleanOptions.HoleFillMaxWidth"/>); it has no effect without <paramref name="clean"/>.</summary>
    public static Mesh Finish(
        Mesh mesh, bool clean = true, double smallComponentAreaRatio = 1e-5, double smallComponentFaceRatio = 0, double openingWidthRatio = 0)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (!(openingWidthRatio >= 0))
            throw new ArgumentOutOfRangeException(nameof(openingWidthRatio), "Ratio must be non-negative.");
        if (clean)
        {
            MeshCleanOptions options = new()
            {
                SmallComponentAreaRatio = smallComponentAreaRatio,
                SmallComponentFaceRatio = smallComponentFaceRatio,
                HoleFillMaxWidth = openingWidthRatio * MeshOps.BoundingBoxDiagonal(mesh.Vertices),
            };
            return MeshOps.ComputeVertexNormals(MeshCleaner.Clean(mesh, options));
        }
        FaceOrientation.Unify(mesh);
        return MeshOps.ComputeVertexNormals(mesh);
    }
}
