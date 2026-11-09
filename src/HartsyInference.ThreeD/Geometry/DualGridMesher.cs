using System.Numerics;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>Triangle mesh produced from dual-grid voxels: <see cref="Vertices"/> <c>[N·3]</c> and <see cref="Triangles"/> <c>[F·3]</c>.</summary>
public sealed record DualGridMesh(float[] Vertices, int[] Triangles);

/// <summary>
/// CPU dual-grid → triangle mesh extraction (TRELLIS.2 O-Voxel <c>flexible_dual_grid_to_mesh</c>, inference path).
/// Each voxel flagged as intersected on an axis emits one quad through its four edge neighbours; the quad is split
/// into two triangles by <c>splitWeight</c>, or by the reference's alignment test when no weights are given.
/// </summary>
public static class DualGridMesher
{
    // Per axis (x, y, z), the four voxel offsets of the quad's corners in winding order.
    private static readonly int[][][] EdgeOffsets =
    {
        new[] { new[] { 0, 0, 0 }, new[] { 0, 0, 1 }, new[] { 0, 1, 1 }, new[] { 0, 1, 0 } },
        new[] { new[] { 0, 0, 0 }, new[] { 1, 0, 0 }, new[] { 1, 0, 1 }, new[] { 0, 0, 1 } },
        new[] { new[] { 0, 0, 0 }, new[] { 0, 1, 0 }, new[] { 1, 1, 0 }, new[] { 1, 0, 0 } },
    };

    // Quad-corner orderings for the two diagonal splits (corner indices into the quad).
    private static readonly int[] Split1 = { 0, 1, 2, 0, 2, 3 };
    private static readonly int[] Split2 = { 0, 1, 3, 3, 1, 2 };

    /// <summary>Extracts the mesh. <paramref name="coords"/> is <c>[N·3]</c> integer voxel coords, <paramref name="dualVertices"/> <c>[N·3]</c> offsets in [0,1]³, <paramref name="intersected"/> <c>[N·3]</c> per-axis flags.</summary>
    /// <param name="splitWeight">Per-voxel <c>[N]</c> split weights, or null to use the reference alignment test.</param>
    public static DualGridMesh Extract(
        int[] coords, float[] dualVertices, bool[] intersected, float[]? splitWeight, Vector3 aabbMin, Vector3 voxelSize)
    {
        ArgumentNullException.ThrowIfNull(coords);
        ArgumentNullException.ThrowIfNull(dualVertices);
        ArgumentNullException.ThrowIfNull(intersected);
        if (coords.Length % 3 != 0) throw new ArgumentException("coords must be [N*3].", nameof(coords));
        int n = coords.Length / 3;
        if (dualVertices.Length != n * 3) throw new ArgumentException("dualVertices must be [N*3].", nameof(dualVertices));
        if (intersected.Length != n * 3) throw new ArgumentException("intersected must be [N*3].", nameof(intersected));
        if (splitWeight is not null && splitWeight.Length != n) throw new ArgumentException("splitWeight must be [N].", nameof(splitWeight));

        float[] vertices = new float[n * 3];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                vertices[i * 3 + c] = (coords[i * 3 + c] + dualVertices[i * 3 + c]) * Component(voxelSize, c) + Component(aabbMin, c);
            }
        }

        Dictionary<(int, int, int), int> index = new(n);
        for (int i = 0; i < n; i++)
        {
            if (!index.TryAdd((coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]), i))
            {
                throw new ArgumentException($"Duplicate voxel coordinate at index {i}.", nameof(coords));
            }
        }

        List<int> quads = new();
        for (int i = 0; i < n; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                if (!intersected[i * 3 + axis]) continue;
                if (TryLookupQuad(coords, index, i, axis, out int q0, out int q1, out int q2, out int q3))
                {
                    quads.Add(q0); quads.Add(q1); quads.Add(q2); quads.Add(q3);
                }
            }
        }

        List<int> triangles = new(quads.Count / 4 * 6);
        for (int q = 0; q < quads.Count; q += 4)
        {
            int[] quad = { quads[q], quads[q + 1], quads[q + 2], quads[q + 3] };
            bool useSplit1 = splitWeight is not null
                ? WeightedSplitIsOne(splitWeight, quad)
                : AlignedSplitIsOne(vertices, quad);
            int[] order = useSplit1 ? Split1 : Split2;
            for (int k = 0; k < order.Length; k++) triangles.Add(quad[order[k]]);
        }

        return new DualGridMesh(vertices, triangles.ToArray());
    }

    private static bool TryLookupQuad(
        int[] coords, Dictionary<(int, int, int), int> index, int voxel, int axis,
        out int q0, out int q1, out int q2, out int q3)
    {
        int[][] offsets = EdgeOffsets[axis];
        int x = coords[voxel * 3], y = coords[voxel * 3 + 1], z = coords[voxel * 3 + 2];
        q0 = q1 = q2 = q3 = -1;
        return index.TryGetValue((x + offsets[0][0], y + offsets[0][1], z + offsets[0][2]), out q0)
            && index.TryGetValue((x + offsets[1][0], y + offsets[1][1], z + offsets[1][2]), out q1)
            && index.TryGetValue((x + offsets[2][0], y + offsets[2][1], z + offsets[2][2]), out q2)
            && index.TryGetValue((x + offsets[3][0], y + offsets[3][1], z + offsets[3][2]), out q3);
    }

    // Reference: w02 = w0*w2, w13 = w1*w3; split 1 when w02 > w13.
    private static bool WeightedSplitIsOne(float[] w, int[] quad)
    {
        float w02 = w[quad[0]] * w[quad[2]];
        float w13 = w[quad[1]] * w[quad[3]];
        return w02 > w13;
    }

    // Reference alignment test, replicated exactly: each split's two normals use the gathered column order of that
    // split's index table. For the diagonal splits this compares a triangle with itself (split 1) and yields a zero
    // normal (split 2), so split 1 wins unless its first triangle is degenerate. TRELLIS.2 passes split weights, so
    // this path is used only when a caller omits them.
    private static bool AlignedSplitIsOne(float[] v, int[] quad)
    {
        float align0 = AlignOf(v, quad, Split1);
        float align1 = AlignOf(v, quad, Split2);
        return align0 > align1;
    }

    private static float AlignOf(float[] v, int[] quad, int[] split)
    {
        Vector3 a = Position(v, quad[split[0]]), b = Position(v, quad[split[1]]);
        Vector3 c = Position(v, quad[split[2]]), d = Position(v, quad[split[3]]);
        Vector3 n0 = Vector3.Cross(b - a, c - a);
        Vector3 n1 = Vector3.Cross(c - b, d - b);
        return MathF.Abs(Vector3.Dot(n0, n1));
    }

    private static Vector3 Position(float[] v, int vertex) => new(v[vertex * 3], v[vertex * 3 + 1], v[vertex * 3 + 2]);

    private static float Component(Vector3 v, int c) => c switch
    {
        0 => v.X,
        1 => v.Y,
        _ => v.Z,
    };
}
