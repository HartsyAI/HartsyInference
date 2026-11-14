namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Quadric-error edge-collapse simplification. Collapses preserve manifold edges and closedness, pin boundary edges with penalty planes,
/// and are refused when they would fold a neighboring face; the result is deterministic for a given input. Output is geometry-only
/// (positions and indices).
/// </summary>
public static class MeshDecimator
{
    /// <summary>How far over its target a staged result may stay before the pinch pass runs: 5 %.</summary>
    private const double StallRatio = 1.05;

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
    /// target, clean again and unify orientation. Cleaning can move the final face count slightly off the target. When the result is more than 5 % over
    /// the target and still has boundary edges, a third pass with <see cref="MeshDecimatorOptions.AllowBoundaryPinch"/> may join boundary loops at a
    /// pinch vertex; <see cref="MeshDecimatorOptions.PinchWhenStalled"/> turns that off.</summary>
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
        MeshDecimatorOptions pass3 = options ?? new MeshDecimatorOptions();
        if (pass3.PinchWhenStalled && !pass3.AllowBoundaryPinch && result.TriangleCount > targetFaceCount * StallRatio && HasBoundary(result))
            result = MeshCleaner.Clean(Decimate(result, targetFaceCount, pass3 with { AllowBoundaryPinch = true }, cancellationToken), clean);
        return result;
    }

    private static bool HasBoundary(Mesh mesh)
    {
        EdgeTable table = EdgeTable.Build(mesh.Indices, mesh.VertexCount);
        for (int start = 0; start < table.Count;)
        {
            int end = table.RunEnd(start);
            if (end - start == 1)
                return true;
            start = end;
        }
        return false;
    }
}
