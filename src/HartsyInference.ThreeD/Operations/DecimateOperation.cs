using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Reduces the triangle count by quadric edge collapse. The target is shared among the primitives in proportion to their size, so a model made of parts keeps every part; the total can land a little above the target where a primitive runs out of legal collapses. Vertices at one position are welded first, so a UV seam does not pin the collapse. The collapse is geometry only: a primitive that is reduced loses its UVs, vertex colours and so its textures (it keeps its material colour), and gets fresh smooth normals if it had normals. A model already within the target is left exactly as it is.</summary>
public sealed class DecimateOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "decimate";

    /// <inheritdoc/>
    public string Description => "Reduce the triangle count to a target (quadric edge collapse). Give triangles or ratio. Every reduced primitive loses its UVs, vertex colours and textures (the material colour stays); a model already within the target is untouched.";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } =
    [
        new("triangles", "The triangle count to reach over the whole model.", null, 1, 100_000_000, Integer: true),
        new("ratio", "The share of triangles to keep, in (0, 1); an alternative to triangles.", null, 0.0001, 0.9999),
    ];

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel)
    {
        if (arguments.Has("triangles") == arguments.Has("ratio"))
            throw new ArgumentException("decimate takes exactly one of triangles or ratio.");
        long total = document.TriangleCount;
        long target = arguments.Has("triangles") ? arguments.GetInt("triangles") : (long)Math.Round(total * arguments.Get("ratio"));
        target = Math.Max(target, 1);
        return PrimitiveEdit.Map(document, Id, (_, primitive) =>
        {
            Mesh g = primitive.Geometry;
            int share = (int)Math.Clamp((long)Math.Round((double)target * g.TriangleCount / Math.Max(total, 1)), 1, int.MaxValue);
            if (g.TriangleCount <= share)
                return (primitive, null);
            Mesh reduced = MeshDecimator.Decimate(WeldedVertices.Build(g).ToMesh(), share, null, cancel);
            if (g.Normals is not null)
                MeshOps.ComputeVertexNormals(reduced);
            // A mesh that runs out of legal collapses (boundaries, folds) stops above its share; say so rather than let the count surprise the caller.
            string stopped = reduced.TriangleCount > share * 1.05 + 2 ? " The share was not reached: no legal collapse was left." : "";
            return (new MeshPrimitive { Geometry = reduced, Material = primitive.Material }, $"A primitive was reduced from {g.TriangleCount} to {reduced.TriangleCount} triangles (its share of the target is {share}).{stopped}");
        }, cancel);
    }
}
