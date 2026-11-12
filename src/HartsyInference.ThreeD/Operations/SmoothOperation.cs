using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Taubin smoothing: each pass moves a vertex towards the average of its neighbours (factor lambda) and the next pass moves it back out (factor mu, slightly larger and negative), so noise goes and the volume does not shrink. Vertices at one position move together, so a UV seam does not tear, and vertices on an open boundary stay where they are. Only positions change (normals are recomputed when the primitive had them); UVs, colours and textures are kept.</summary>
public sealed class SmoothOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "smooth";

    /// <inheritdoc/>
    public string Description => "Taubin smoothing that removes surface noise without shrinking the model. Open-boundary vertices stay fixed. Keeps UVs, colours and textures.";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } =
    [
        new("iterations", "Smoothing passes (each is a shrink step and an inflate step).", 5, 1, 200, Integer: true),
        new("lambda", "The step towards the neighbour average; larger smooths faster.", 0.5, 0.01, 1),
    ];

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel)
    {
        int iterations = arguments.GetInt("iterations");
        float lambda = (float)arguments.Get("lambda");
        // The classic Taubin pair: 1/lambda + 1/mu is about 0.1, which passes low frequencies and removes high ones.
        float mu = 1f / (0.1f - 1f / lambda);
        return PrimitiveEdit.Map(document, Id, (_, primitive) =>
        {
            Mesh g = primitive.Geometry;
            WeldedVertices welded = WeldedVertices.Build(g);
            float[] smoothed = Smooth(welded, iterations, lambda, mu, cancel);
            float[] positions = new float[g.Vertices.Length];
            for (int v = 0; v < welded.Group.Length; v++)
                Array.Copy(smoothed, welded.Group[v] * 3, positions, v * 3, 3);
            Mesh moved = PrimitiveEdit.WithPositions(g, positions);
            if (g.Normals is not null)
                moved.Normals = PrimitiveEdit.WeldedNormals(moved, WeldedVertices.Build(moved));
            return (new MeshPrimitive { Geometry = moved, Material = primitive.Material }, null);
        }, cancel);
    }

    private static float[] Smooth(WeldedVertices welded, int iterations, float lambda, float mu, CancellationToken cancel)
    {
        int n = welded.Count;
        Dictionary<long, int> edges = [];
        for (int t = 0; t < welded.Indices.Length; t += 3)
        {
            for (int k = 0; k < 3; k++)
            {
                int a = welded.Indices[t + k], b = welded.Indices[t + (k + 1) % 3];
                if (a == b)
                    continue;
                // lo * n + hi, not a packed pair: a long hashes as the xor of its halves, so packed pairs collide by the thousand on a real mesh.
                long key = a < b ? (long)a * n + b : (long)b * n + a;
                edges[key] = edges.GetValueOrDefault(key) + 1;
            }
        }
        int[] degree = new int[n];
        bool[] fixedVertex = new bool[n];
        foreach ((long key, int uses) in edges)
        {
            int a = (int)(key / n), b = (int)(key % n);
            degree[a]++;
            degree[b]++;
            // An edge used by one triangle lies on an open boundary.
            if (uses == 1)
            {
                fixedVertex[a] = true;
                fixedVertex[b] = true;
            }
        }
        int[] start = new int[n + 1];
        for (int i = 0; i < n; i++)
            start[i + 1] = start[i] + degree[i];
        int[] neighbours = new int[start[n]];
        int[] fill = new int[n];
        foreach (long key in edges.Keys)
        {
            int a = (int)(key / n), b = (int)(key % n);
            neighbours[start[a] + fill[a]++] = b;
            neighbours[start[b] + fill[b]++] = a;
        }

        float[] current = (float[])welded.Positions.Clone(), next = new float[current.Length];
        for (int pass = 0; pass < iterations * 2; pass++)
        {
            cancel.ThrowIfCancellationRequested();
            float factor = pass % 2 == 0 ? lambda : mu;
            for (int v = 0; v < n; v++)
            {
                int count = start[v + 1] - start[v];
                if (fixedVertex[v] || count == 0)
                {
                    Array.Copy(current, v * 3, next, v * 3, 3);
                    continue;
                }
                float sx = 0, sy = 0, sz = 0;
                for (int k = start[v]; k < start[v + 1]; k++)
                {
                    int w = neighbours[k];
                    sx += current[w * 3];
                    sy += current[w * 3 + 1];
                    sz += current[w * 3 + 2];
                }
                next[v * 3] = current[v * 3] + factor * (sx / count - current[v * 3]);
                next[v * 3 + 1] = current[v * 3 + 1] + factor * (sy / count - current[v * 3 + 1]);
                next[v * 3 + 2] = current[v * 3 + 2] + factor * (sz / count - current[v * 3 + 2]);
            }
            (current, next) = (next, current);
        }
        return current;
    }
}
