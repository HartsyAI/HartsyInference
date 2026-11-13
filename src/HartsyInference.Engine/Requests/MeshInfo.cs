namespace HartsyInference.Engine.Requests;

/// <summary>What a host needs to know about a result without opening it: its size, geometry, topology and contents.</summary>
public sealed record MeshInfo
{
    /// <summary>Bytes of <see cref="MeshResult.Data"/>.</summary>
    public required long ByteSize { get; init; }

    /// <summary>Triangles of a mesh result; zero for splats.</summary>
    public long TriangleCount { get; init; }

    /// <summary>Vertices of a mesh result as stored (a UV seam duplicates one); zero for splats.</summary>
    public long VertexCount { get; init; }

    /// <summary>Gaussian splats of a splat result; zero for a mesh.</summary>
    public long SplatCount { get; init; }

    /// <summary>Smallest <c>[x, y, z]</c> of the geometry, or null when it could not be measured.</summary>
    public float[]? BoundsMin { get; init; }

    /// <summary>Largest <c>[x, y, z]</c> of the geometry, or null when it could not be measured.</summary>
    public float[]? BoundsMax { get; init; }

    /// <summary>Vertex-connected pieces of the welded surface; null when the topology was not measured.</summary>
    public int? Components { get; init; }

    /// <summary>Edges of the welded surface with one face; null when not measured.</summary>
    public long? BoundaryEdges { get; init; }

    /// <summary>Edges of the welded surface with more than two faces; null when not measured.</summary>
    public long? NonManifoldEdges { get; init; }

    /// <summary>The surface is closed and every edge has two faces; null when not measured.</summary>
    public bool? Watertight { get; init; }

    /// <summary>Some primitive carries normals.</summary>
    public bool HasNormals { get; init; }

    /// <summary>The file has a texture.</summary>
    public bool HasTexture { get; init; }

    /// <summary>The file has a skin.</summary>
    public bool HasSkin { get; init; }

    /// <summary>Names of the file's animations.</summary>
    public IReadOnlyList<string> AnimationNames { get; init; } = [];
}
