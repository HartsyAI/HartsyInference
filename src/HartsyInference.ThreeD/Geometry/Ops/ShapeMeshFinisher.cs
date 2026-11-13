namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>The last step of a shape-only mesh from a dual-grid decoder (TRELLIS.2, Pixal3D): the repair the textured path gets from
/// <see cref="PbrMeshPostProcessor"/>, without the decimation and the bake. The raw mesh has non-manifold edges and loose specks; the repair
/// welds, drops degenerate and duplicate faces, splits non-manifold edges, removes tiny components, closes small holes and unifies the winding
/// (<see cref="MeshCleanOptions"/> defaults). It does not make the mesh watertight: boundary edges along larger openings remain, which upstream
/// closes with a dual-contouring remesh this engine does not have.</summary>
public static class ShapeMeshFinisher
{
    /// <summary>Returns the mesh to export: repaired with <see cref="MeshCleaner.Clean"/> when <paramref name="clean"/> is set, otherwise
    /// the winding unified only, then smooth vertex normals. The input is not modified when it is repaired. <paramref name="smallComponentAreaRatio"/> is
    /// <see cref="MeshCleanOptions.SmallComponentAreaRatio"/>: a model whose extraction leaves larger floating pieces raises it. <paramref name="smallComponentFaceRatio"/> is
    /// <see cref="MeshCleanOptions.SmallComponentFaceRatio"/>, the rule of a model whose upstream removes floaters by face count (the Hunyuan3D shape models).</summary>
    public static Mesh Finish(Mesh mesh, bool clean = true, double smallComponentAreaRatio = 1e-5, double smallComponentFaceRatio = 0)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (clean)
            return MeshOps.ComputeVertexNormals(MeshCleaner.Clean(mesh, new MeshCleanOptions { SmallComponentAreaRatio = smallComponentAreaRatio, SmallComponentFaceRatio = smallComponentFaceRatio }));
        FaceOrientation.Unify(mesh);
        return MeshOps.ComputeVertexNormals(mesh);
    }
}
