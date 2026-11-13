using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Operations;

/// <summary>A mesh's vertices grouped by position. A textured mesh stores a vertex once per UV island or normal, so two vertices at one point belong to one surface point; topology (neighbours, holes, components) and smoothing have to see them as one, and this is the map that does.</summary>
internal sealed class WeldedVertices
{
    private WeldedVertices(int[] group, float[] positions, int[] indices)
    {
        Group = group;
        Positions = positions;
        Indices = indices;
    }

    /// <summary>The group of each original vertex.</summary>
    public int[] Group { get; }

    /// <summary>The position of each group, <c>3 * Count</c> floats.</summary>
    public float[] Positions { get; }

    /// <summary>The triangle indices in terms of groups.</summary>
    public int[] Indices { get; }

    /// <summary>The number of groups.</summary>
    public int Count => Positions.Length / 3;

    /// <summary>Groups the vertices of <paramref name="mesh"/> whose positions are bit-for-bit equal (minus and plus zero count as one).</summary>
    public static WeldedVertices Build(Mesh mesh)
    {
        int n = mesh.VertexCount;
        int[] group = new int[n];
        Dictionary<(int, int, int), int> seen = new(n);
        List<float> positions = new(n * 3);
        for (int i = 0; i < n; i++)
        {
            float x = mesh.Vertices[i * 3] + 0f, y = mesh.Vertices[i * 3 + 1] + 0f, z = mesh.Vertices[i * 3 + 2] + 0f;
            (int, int, int) key = (BitConverter.SingleToInt32Bits(x), BitConverter.SingleToInt32Bits(y), BitConverter.SingleToInt32Bits(z));
            if (!seen.TryGetValue(key, out int id))
            {
                id = positions.Count / 3;
                seen[key] = id;
                positions.Add(x);
                positions.Add(y);
                positions.Add(z);
            }
            group[i] = id;
        }
        int[] indices = new int[mesh.Indices.Length];
        for (int i = 0; i < indices.Length; i++)
            indices[i] = group[mesh.Indices[i]];
        return new WeldedVertices(group, [.. positions], indices);
    }

    /// <summary>The welded geometry as a mesh (positions and indices only).</summary>
    public Mesh ToMesh() => new() { Vertices = Positions, Indices = Indices };
}
