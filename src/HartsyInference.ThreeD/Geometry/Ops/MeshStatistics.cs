namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Counts and topology of a triangle mesh, taken on the surface: vertices at the same position are one vertex, so UV seams and split normals do not read as boundaries.</summary>
public static class MeshStatistics
{
    /// <summary>Measures <paramref name="mesh"/>. <see cref="MeshStats.VertexCount"/> and <see cref="MeshStats.TriangleCount"/> are the mesh's own; the edge and component counts are of its welded surface.</summary>
    public static MeshStats Measure(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        float[] min = [float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity];
        float[] max = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
        for (int i = 0; i < mesh.Vertices.Length; i += 3)
        {
            for (int c = 0; c < 3; c++)
            {
                min[c] = Math.Min(min[c], mesh.Vertices[i + c]);
                max[c] = Math.Max(max[c], mesh.Vertices[i + c]);
            }
        }
        if (mesh.VertexCount == 0)
        {
            min = [0, 0, 0];
            max = [0, 0, 0];
        }

        Mesh surface = MeshCleaner.WeldVertices(mesh);
        EdgeTable edges = EdgeTable.Build(surface.Indices, surface.VertexCount);
        long boundary = 0, nonManifold = 0;
        for (int start = 0; start < edges.Count;)
        {
            int end = edges.RunEnd(start);
            int run = end - start;
            if (run == 1)
                boundary++;
            else if (run > 2)
                nonManifold++;
            start = end;
        }

        int components = CountComponents(surface);
        return new MeshStats(mesh.VertexCount, mesh.TriangleCount, min, max, components, boundary, nonManifold,
            mesh.TriangleCount > 0 && boundary == 0 && nonManifold == 0);
    }

    private static int CountComponents(Mesh surface)
    {
        int[] parent = new int[surface.VertexCount];
        for (int i = 0; i < parent.Length; i++)
            parent[i] = i;
        for (int t = 0; t < surface.Indices.Length; t += 3)
        {
            Union(parent, surface.Indices[t], surface.Indices[t + 1]);
            Union(parent, surface.Indices[t], surface.Indices[t + 2]);
        }
        HashSet<int> roots = [];
        for (int t = 0; t < surface.Indices.Length; t++)
            roots.Add(Find(parent, surface.Indices[t]));
        return roots.Count;
    }

    private static int Find(int[] parent, int index)
    {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        return index;
    }

    private static void Union(int[] parent, int a, int b)
    {
        int rootA = Find(parent, a);
        int rootB = Find(parent, b);
        if (rootA != rootB)
            parent[rootB] = rootA;
    }
}
