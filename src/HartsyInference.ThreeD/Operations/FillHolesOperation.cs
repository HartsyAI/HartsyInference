using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Closes boundary loops with a triangle or a fan around a new centre vertex. Vertices at one position are welded first, so a UV seam is not mistaken for a hole. The result is geometry only: a primitive that gains triangles loses its UVs, vertex colours and so its textures (it keeps its material colour); one with no hole to fill is left exactly as it is.</summary>
public sealed class FillHolesOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "fill-holes";

    /// <inheritdoc/>
    public string Description => "Close holes whose perimeter is at most a share of the primitive's bounding-box diagonal. A primitive that is filled loses its UVs, vertex colours and textures; one with no hole is untouched.";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } =
    [
        new("maxPerimeter", "The largest hole to close, as a share of the bounding-box diagonal of its primitive.", 0.05, 0, 10),
    ];

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel)
    {
        double share = arguments.Get("maxPerimeter");
        return PrimitiveEdit.Map(document, Id, (index, primitive) =>
        {
            Mesh g = primitive.Geometry;
            WeldedVertices welded = WeldedVertices.Build(g);
            Mesh weldedMesh = welded.ToMesh();
            Mesh filled = MeshCleaner.FillHoles(weldedMesh, share * PrimitiveEdit.Diagonal(welded.Positions));
            if (filled.TriangleCount == weldedMesh.TriangleCount)
                return (primitive, null);
            if (g.Normals is not null)
                MeshOps.ComputeVertexNormals(filled);
            return (new MeshPrimitive { Geometry = filled, Material = primitive.Material }, $"Primitive {index}: holes were closed, {filled.TriangleCount - weldedMesh.TriangleCount} triangle(s) added.");
        }, cancel);
    }
}
