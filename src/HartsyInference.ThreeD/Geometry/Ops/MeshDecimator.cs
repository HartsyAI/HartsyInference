namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Quadric-error edge-collapse simplification. Collapses preserve manifoldness and closedness, pin boundary edges with penalty planes,
/// and are refused when they would fold a neighboring face; the result is deterministic for a given input. Output is geometry-only
/// (positions and indices).
/// </summary>
public static class MeshDecimator
{
    private const double StallTolerance = 1.05;

    /// <summary>Simplifies <paramref name="mesh"/> toward <paramref name="targetFaceCount"/> faces. Closed manifold meshes land on the target
    /// or one face below; meshes that run out of legal collapses stop early.</summary>
    public static Mesh Decimate(Mesh mesh, int targetFaceCount, MeshDecimatorOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (targetFaceCount < 1)
            throw new ArgumentOutOfRangeException(nameof(targetFaceCount), "Target face count must be positive.");
        if (mesh.Vertices.Length % 3 != 0 || mesh.Indices.Length % 3 != 0)
            throw new ArgumentException("Mesh arrays must hold whole vertices and triangles.", nameof(mesh));
        foreach (int i in mesh.Indices)
            if ((uint)i >= (uint)mesh.VertexCount)
                throw new ArgumentException("A triangle index lies outside the vertex array.", nameof(mesh));
        if (mesh.TriangleCount <= targetFaceCount)
            return new Mesh { Vertices = (float[])mesh.Vertices.Clone(), Indices = (int[])mesh.Indices.Clone() };

        QuadricEdgeCollapser collapser = new(mesh, options ?? new MeshDecimatorOptions());
        collapser.Run(targetFaceCount, cancellationToken);
        return collapser.ToMesh();
    }

    /// <summary>The upstream two-pass schedule: simplify to <paramref name="intermediateFactor"/> times the target, clean, simplify to the
    /// target, clean again and unify orientation. Cleaning can move the final face count slightly off the target.</summary>
    public static Mesh DecimateStaged(Mesh mesh, int targetFaceCount, int intermediateFactor = 3, MeshCleanOptions? cleanOptions = null,
        MeshDecimatorOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (intermediateFactor < 1)
            throw new ArgumentOutOfRangeException(nameof(intermediateFactor), "The intermediate factor must be at least 1.");
        MeshCleanOptions clean = cleanOptions ?? new MeshCleanOptions();
        MeshCleanOptions interim = clean with { UnifyOrientation = false };
        long intermediate = Math.Min((long)targetFaceCount * intermediateFactor, int.MaxValue);

        Mesh pass = Decimate(mesh, (int)intermediate, options, cancellationToken);
        pass = MeshCleaner.Clean(pass, interim);
        cancellationToken.ThrowIfCancellationRequested();
        pass = Decimate(pass, targetFaceCount, options, cancellationToken);
        Mesh result = MeshCleaner.Clean(pass, clean);
        // A mesh torn into many pieces stops above the target because its boundary vertices cannot meet; one more pass may join them.
        if (result.TriangleCount > targetFaceCount * StallTolerance && options?.AllowBoundaryPinch != true)
        {
            MeshDecimatorOptions relaxed = (options ?? new MeshDecimatorOptions()) with { AllowBoundaryPinch = true };
            result = MeshCleaner.Clean(Decimate(result, targetFaceCount, relaxed, cancellationToken), clean);
        }
        return result;
    }
}
