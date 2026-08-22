namespace HartsyInference.ThreeD.Geometry;

/// <summary>A triangle mesh in object space: flat interleaved vertex/index arrays produced by the mesh pipelines (Hunyuan3D, TripoSR), with normals filled in later by <see cref="Ops.MeshOps.ComputeVertexNormals"/>.</summary>
public sealed class Mesh
{
    /// <summary>Vertex positions, length <c>3 * VertexCount</c> (x,y,z interleaved).</summary>
    public required float[] Vertices { get; init; }

    /// <summary>Triangle vertex indices, length <c>3 * TriangleCount</c> (CCW winding).</summary>
    public required int[] Indices { get; init; }

    /// <summary>Per-vertex normals, length <c>3 * VertexCount</c>, or null.</summary>
    public float[]? Normals { get; set; }

    /// <summary>Per-vertex UVs, length <c>2 * VertexCount</c>, or null.</summary>
    public float[]? Uvs { get; set; }

    /// <summary>Per-vertex RGB colors in [0,1], length <c>3 * VertexCount</c>, or null.</summary>
    public float[]? VertexColors { get; set; }

    /// <summary>Number of vertices.</summary>
    public int VertexCount => Vertices.Length / 3;

    /// <summary>Number of triangles.</summary>
    public int TriangleCount => Indices.Length / 3;
}
