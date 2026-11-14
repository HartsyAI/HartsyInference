using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Rebuilds every primitive as a clean closed shell on a regular grid (dual contouring of the offset surface, upstream TRELLIS.2's remesh), which closes the holes and slits a torn surface has. The shell is the surface offset outwards and inwards by band voxels, so a thin feature thickens and, unless outerOnly keeps just the outer sheet, the result holds two sheets. The result is geometry only: it loses UVs, vertex colours and textures (the material colour stays).</summary>
public sealed class RemeshOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "remesh";

    /// <inheritdoc/>
    public string Description => "Rebuild the surface as a clean closed shell on a grid (narrow-band dual contouring, as upstream TRELLIS.2's remesh): holes and slits narrower than twice the band close. Every rebuilt primitive loses its UVs, vertex colours and textures (the material colour stays).";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } =
    [
        new("resolution", "Grid cells along the longest side (512, 1024, 1536 and the like: a number that halves evenly down to a base grid of at most 64 cells); the faces and the memory grow with its square, about 2 GiB at 512 for a 1.5 to 3.8 M-face input.", 256, 32, 1024, Integer: true),
        new("band", "Half the shell's thickness in voxels.", 1, 0.1, 8),
        new("projectBack", "How far each vertex moves back towards the input surface, 0 to 1; close to 1 with outerOnly 0 puts both sheets on the input as coincident surfaces.", 0, 0, 1),
        new("outerOnly", "1 keeps only the outer sheet of the shell (found by flood fill), 0 keeps both.", 0, 0, 1, Integer: true),
    ];

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel)
    {
        NarrowBandRemeshOptions options = new()
        {
            Resolution = arguments.GetInt("resolution"),
            Band = arguments.Get("band"),
            ProjectBack = arguments.Get("projectBack"),
            OuterShellOnly = arguments.GetInt("outerOnly") == 1,
        };
        return PrimitiveEdit.Map(document, Id, (index, primitive) =>
        {
            Mesh input = WeldedVertices.Build(primitive.Geometry).ToMesh();
            if (input.TriangleCount == 0)
                return (primitive, null);
            Mesh remeshed = NarrowBandRemesher.Remesh(input, options, null, cancel);
            if (primitive.Geometry.Normals is not null)
                MeshOps.ComputeVertexNormals(remeshed);
            return (new MeshPrimitive { Geometry = remeshed, Material = primitive.Material },
                $"Primitive {index}: rebuilt at {options.Resolution} cells as {remeshed.TriangleCount} triangles (it had {primitive.Geometry.TriangleCount}).");
        }, cancel);
    }
}
