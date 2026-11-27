using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Replaces every primitive's normals with smooth ones: the area-weighted normal of the faces around each surface point, shared by the vertices at one position, so a UV seam does not show as a crease. Nothing else changes; UVs, colours and textures are kept.</summary>
public sealed class RecomputeNormalsOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "recompute-normals";

    /// <inheritdoc/>
    public string Description => "Recompute smooth vertex normals (shared across UV seams). Keeps everything else.";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } = [];

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel) =>
        PrimitiveEdit.Map(document, Id, (_, primitive) =>
        {
            Mesh g = primitive.Geometry;
            Mesh result = new() { Vertices = g.Vertices, Indices = g.Indices, Uvs = g.Uvs, VertexColors = g.VertexColors, Normals = PrimitiveEdit.WeldedNormals(g, WeldedVertices.Build(g)) };
            return (new MeshPrimitive { Geometry = result, Material = primitive.Material }, null);
        }, cancel);
}
