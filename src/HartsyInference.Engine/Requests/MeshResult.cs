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

    /// <summary>A decimated glb of <see cref="Data"/> for a viewer when the request set <see cref="MeshRequest.PreviewTriangles"/> and the mesh is larger; geometry only (no texture or skin), in the same frame as <see cref="Data"/>. Null when no preview was asked for, or when <see cref="Data"/> is already within the budget.</summary>
    public byte[]? Preview { get; init; }

    /// <summary>Size, geometry, topology and contents of <see cref="Data"/>; null for a result the engine did not measure.</summary>
    public MeshInfo? Info { get; init; }

    /// <summary>The frame <see cref="Data"/> is in: what the request asked for, as applied.</summary>
    public MeshFrame Frame { get; init; }

    /// <summary>Things a host should tell its user: a frame that could not be fully normalised, topology that could not be measured.</summary>
    public IReadOnlyList<string> Warnings { get; init; } = [];
}

/// <summary>One part found by a segmentation model.</summary>
/// <param name="Id">The part label.</param>
/// <param name="Box"><c>[minX, minY, minZ, maxX, maxY, maxZ]</c> in the units of the mesh the model segmented.</param>
/// <param name="FaceCount">Triangles of the part.</param>
public sealed record MeshPart(int Id, IReadOnlyList<float> Box, int FaceCount);
