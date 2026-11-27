using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Removes floating pieces: connected components (through shared surface points, so a UV seam does not split one) whose area is below a share of their primitive's total area; the largest always stays. The rule is <see cref="MeshCleaner"/>'s, applied to the welded copy of the primitive; the cut itself keeps every vertex attribute, so UVs, normals, vertex colours and textures survive for what remains.</summary>
public sealed class RemoveSmallComponentsOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "remove-small-components";

    /// <inheritdoc/>
    public string Description => "Remove floating pieces smaller than a share of the primitive's surface area (the largest piece always stays). Keeps UVs, normals, colours and textures.";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } =
    [
        new("areaRatio", "The share of the primitive's total area below which a piece is removed.", 0.001, 0, 1),
    ];

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel)
    {
        double ratio = arguments.Get("areaRatio");
        return PrimitiveEdit.Map(document, Id, (index, primitive) =>
        {
            Mesh g = primitive.Geometry;
            bool[] keep = MeshCleaner.SmallComponentFaceMask(WeldedVertices.Build(g).ToMesh(), ratio);
            int kept = keep.Count(k => k);
            if (kept == keep.Length)
                return (primitive, null);
            return (new MeshPrimitive { Geometry = Subset(g, keep), Material = primitive.Material }, $"Primitive {index}: removed {keep.Length - kept} triangle(s) of floating pieces.");
        }, cancel);
    }

    /// <summary>The kept faces with the vertices they use, in their original order, carrying every attribute array along.</summary>
    private static Mesh Subset(Mesh mesh, bool[] keep)
    {
        int[] remap = new int[mesh.VertexCount];
        Array.Fill(remap, -1);
        List<int> indices = [];
        int used = 0;
        for (int f = 0; f < keep.Length; f++)
        {
            if (!keep[f])
                continue;
            for (int k = 0; k < 3; k++)
            {
                int v = mesh.Indices[f * 3 + k];
                if (remap[v] < 0)
                    remap[v] = used++;
                indices.Add(remap[v]);
            }
        }
        float[] vertices = new float[used * 3];
        float[]? normals = mesh.Normals is null ? null : new float[used * 3];
        float[]? uvs = mesh.Uvs is null ? null : new float[used * 2];
        float[]? colours = mesh.VertexColors is null ? null : new float[used * 3];
        for (int v = 0; v < remap.Length; v++)
        {
            int to = remap[v];
            if (to < 0)
                continue;
            Array.Copy(mesh.Vertices, v * 3, vertices, to * 3, 3);
            if (normals is not null)
                Array.Copy(mesh.Normals!, v * 3, normals, to * 3, 3);
            if (uvs is not null)
                Array.Copy(mesh.Uvs!, v * 2, uvs, to * 2, 2);
            if (colours is not null)
                Array.Copy(mesh.VertexColors!, v * 3, colours, to * 3, 3);
        }
        return new Mesh { Vertices = vertices, Indices = [.. indices], Normals = normals, Uvs = uvs, VertexColors = colours };
    }
}
