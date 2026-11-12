using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Removes floating pieces: connected components (through shared surface points, so a UV seam does not split one) whose area is below a share of their primitive's total area; the largest always stays. Nothing else about the primitive changes: UVs, normals, vertex colours and textures are kept for what remains.</summary>
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
        return PrimitiveEdit.Map(document, Id, (_, primitive) =>
        {
            Mesh g = primitive.Geometry;
            WeldedVertices welded = WeldedVertices.Build(g);
            int faces = g.TriangleCount;
            int[] parent = new int[welded.Count];
            for (int i = 0; i < parent.Length; i++)
                parent[i] = i;
            for (int f = 0; f < faces; f++)
            {
                Union(parent, welded.Indices[f * 3], welded.Indices[f * 3 + 1]);
                Union(parent, welded.Indices[f * 3], welded.Indices[f * 3 + 2]);
            }
            double[] area = new double[welded.Count];
            double total = 0;
            for (int f = 0; f < faces; f++)
            {
                System.Numerics.Vector3 a = PrimitiveEdit.At(welded.Positions, welded.Indices[f * 3]);
                double face = System.Numerics.Vector3.Cross(PrimitiveEdit.At(welded.Positions, welded.Indices[f * 3 + 1]) - a, PrimitiveEdit.At(welded.Positions, welded.Indices[f * 3 + 2]) - a).Length() * 0.5;
                area[Find(parent, welded.Indices[f * 3])] += face;
                total += face;
            }
            double largest = area.Max();
            bool[] keep = new bool[faces];
            int kept = 0;
            HashSet<int> pieces = [], removedPieces = [];
            for (int f = 0; f < faces; f++)
            {
                int root = Find(parent, welded.Indices[f * 3]);
                pieces.Add(root);
                keep[f] = area[root] >= largest || area[root] >= ratio * total;
                kept += keep[f] ? 1 : 0;
                if (!keep[f])
                    removedPieces.Add(root);
            }
            if (kept == faces)
                return (primitive, null);
            return (new MeshPrimitive { Geometry = Subset(g, keep), Material = primitive.Material },
                $"Removed {removedPieces.Count} of {pieces.Count} piece(s) in a primitive ({faces - kept} triangles).");
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

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private static void Union(int[] parent, int a, int b)
    {
        int ra = Find(parent, a), rb = Find(parent, b);
        if (ra != rb)
            parent[ra] = rb;
    }
}
