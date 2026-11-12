using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Document;

/// <summary>One triangle batch of a document: its geometry, in world space, and the material it uses.</summary>
public sealed class MeshPrimitive
{
    /// <summary>Positions, indices and the optional normals, UVs (glTF convention: origin top-left) and vertex colours.</summary>
    public required Mesh Geometry { get; init; }

    /// <summary>Index into <see cref="MeshDocument.Materials"/>, or -1 for none.</summary>
    public int Material { get; init; } = -1;
}
