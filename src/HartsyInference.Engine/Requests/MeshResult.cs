namespace HartsyInference.Engine.Requests;

/// <summary>The result of a 3D generation: the encoded mesh bytes and their container format.</summary>
public sealed record MeshResult
{
    /// <summary>Encoded mesh bytes.</summary>
    public required byte[] Data { get; init; }

    /// <summary>Container/format of <see cref="Data"/> (e.g. "glb").</summary>
    public string Format { get; init; } = "glb";

    /// <summary>The parts a segmentation model found (null for every other model): the faces of <see cref="Data"/> are split into one node per part.</summary>
    public IReadOnlyList<MeshPart>? Parts { get; init; }
}

/// <summary>One part found by a segmentation model.</summary>
/// <param name="Id">The part label.</param>
/// <param name="Box"><c>[minX, minY, minZ, maxX, maxY, maxZ]</c> in the units of the mesh the model segmented.</param>
/// <param name="FaceCount">Triangles of the part.</param>
public sealed record MeshPart(int Id, IReadOnlyList<float> Box, int FaceCount);
