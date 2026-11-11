namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>A welded mesh normalized into [-1, 1]³ with unit face normals and angle-weighted vertex normals, ready for sampling.</summary>
public sealed record UniRigPreparedMesh
{
    /// <summary>Normalized positions, length <c>3 * VertexCount</c>.</summary>
    public required float[] Vertices { get; init; }

    /// <summary>Triangle indices, length <c>3 * FaceCount</c>.</summary>
    public required int[] Faces { get; init; }

    /// <summary>Unit vertex normals, length <c>3 * VertexCount</c>.</summary>
    public required float[] VertexNormals { get; init; }

    /// <summary>Unit face normals, length <c>3 * FaceCount</c>.</summary>
    public required float[] FaceNormals { get; init; }

    /// <summary>For each vertex of the source mesh, its vertex here after welding, or <c>-1</c> when the source vertex was dropped as non-finite.</summary>
    public required int[] SourceVertex { get; init; }

    /// <summary>Centre of the source bounding box: <c>source = normalized * Scale + Center</c>.</summary>
    public required float[] Center { get; init; }

    /// <summary>Half of the source bounding box's longest side.</summary>
    public required float Scale { get; init; }

    public int VertexCount => Vertices.Length / 3;

    public int FaceCount => Faces.Length / 3;

    /// <summary>Maps a point from the normalized cube back to source coordinates.</summary>
    public float[] ToSource(ReadOnlySpan<float> normalized)
    {
        float[] result = new float[normalized.Length];
        for (int i = 0; i < normalized.Length; i++)
        {
            result[i] = normalized[i] * Scale + Center[i % 3];
        }

        return result;
    }
}
