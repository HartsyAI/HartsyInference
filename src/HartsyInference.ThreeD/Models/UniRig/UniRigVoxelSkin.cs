using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// The geometric skin prior of UniRig's skin stage: shortest paths from each joint over a graph of mesh edges plus the
/// interior voxels of the mesh, turned into per-vertex weights by an inverse-distance formula. The interior comes from
/// first/last surface hits along each axis (two of three axes must agree); upstream renders six depth views for the
/// same purpose.
/// </summary>
public static class UniRigVoxelSkin
{
    /// <summary>Weights <c>[N, J]</c> (row-major, each row sums to 1) of every vertex for every joint.</summary>
    /// <param name="vertices">Positions, length <c>3 * N</c>, normalized into [-1, 1]³ by their own bounding box.</param>
    /// <param name="faces">Triangle indices, length <c>3 * F</c>.</param>
    /// <param name="joints">Joint positions in the same frame, length <c>3 * J</c>.</param>
    public static float[] Compute(float[] vertices, int[] faces, float[] joints, UniRigVoxelSkinOptions options, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(faces);
        ArgumentNullException.ThrowIfNull(joints);
        ArgumentNullException.ThrowIfNull(options);
        int n = vertices.Length / 3;
        int j = joints.Length / 3;
        if (n == 0 || faces.Length == 0 || j == 0 || options.Grid < 2)
        {
            throw new HartsyInferenceException("UniRig voxel skin needs vertices, faces, at least one joint and a grid of at least 2.");
        }

        int grid = options.Grid;
        float cell = 2f / grid;
        List<int> interior = Interior(vertices, faces, grid);
        int m = interior.Count / 3;
        if ((long)grid * grid * grid > int.MaxValue)
        {
            throw new HartsyInferenceException("UniRig voxel grid is too large.");
        }

        int[] cellNode = new int[grid * grid * grid];
        Array.Fill(cellNode, -1);
        float[] centers = new float[3 * m];
        for (int c = 0; c < m; c++)
        {
            int x = interior[3 * c];
            int y = interior[3 * c + 1];
            int z = interior[3 * c + 2];
            cellNode[(x * grid + y) * grid + z] = n + c;
            centers[3 * c] = (x + 0.5f) * cell - 1f;
            centers[3 * c + 1] = (y + 0.5f) * cell - 1f;
            centers[3 * c + 2] = (z + 0.5f) * cell - 1f;
        }

        Graph graph = BuildGraph(vertices, faces, n, m, grid, cell, cellNode, centers, options);
        UniRigKdTree vertexTree = new(vertices);
        int[] sources = new int[j];
        for (int joint = 0; joint < j; joint++)
        {
            sources[joint] = NearestNode(vertexTree, centers, m, n, joints[3 * joint], joints[3 * joint + 1], joints[3 * joint + 2]);
        }

        float[][] distance = new float[j][];
        Parallel.For(0, j, new ParallelOptions { CancellationToken = cancel }, joint => distance[joint] = Dijkstra(graph, sources[joint], n));
        return Weights(vertices, joints, distance, n, j, options.Alpha);
    }

    /// <summary>Interior voxel cells as flat <c>(x, y, z)</c> index triples: inside on at least two of the three axes.</summary>
    internal static List<int> Interior(float[] vertices, int[] faces, int grid)
    {
        float[][] low = new float[3][];
        float[][] high = new float[3][];
        for (int axis = 0; axis < 3; axis++)
        {
            low[axis] = new float[grid * grid];
            high[axis] = new float[grid * grid];
            Array.Fill(low[axis], float.PositiveInfinity);
            Array.Fill(high[axis], float.NegativeInfinity);
            Rasterize(vertices, faces, grid, axis, low[axis], high[axis]);
        }

        float cell = 2f / grid;
        List<int> interior = [];
        for (int x = 0; x < grid; x++)
        {
            float cx = (x + 0.5f) * cell - 1f;
            for (int y = 0; y < grid; y++)
            {
                float cy = (y + 0.5f) * cell - 1f;
                for (int z = 0; z < grid; z++)
                {
                    float cz = (z + 0.5f) * cell - 1f;
                    int votes = 0;
                    votes += Between(low[0][y * grid + z], high[0][y * grid + z], cx) ? 1 : 0;
                    votes += Between(low[1][x * grid + z], high[1][x * grid + z], cy) ? 1 : 0;
                    votes += Between(low[2][x * grid + y], high[2][x * grid + y], cz) ? 1 : 0;
                    if (votes >= 2)
                    {
                        interior.Add(x);
                        interior.Add(y);
                        interior.Add(z);
                    }
                }
            }
        }

        return interior;
    }

    private static bool Between(float low, float high, float value) => low <= value && value <= high;

    /// <summary>Records, per pixel of an orthographic view along <paramref name="axis"/>, the nearest and farthest surface coordinate along that axis.</summary>
    private static void Rasterize(float[] v, int[] faces, int grid, int axis, float[] low, float[] high)
    {
        int u = axis == 0 ? 1 : 0;
        int w = axis == 2 ? 1 : 2;
        float cell = 2f / grid;
        for (int f = 0; f < faces.Length / 3; f++)
        {
            int a = 3 * faces[3 * f];
            int b = 3 * faces[3 * f + 1];
            int c = 3 * faces[3 * f + 2];
            float au = v[a + u], av = v[a + w], ad = v[a + axis];
            float bu = v[b + u], bv = v[b + w], bd = v[b + axis];
            float cu = v[c + u], cv = v[c + w], cd = v[c + axis];
            float det = (bu - au) * (cv - av) - (cu - au) * (bv - av);
            if (MathF.Abs(det) < 1e-20f)
            {
                continue;
            }

            int u0 = Math.Max(0, (int)MathF.Floor((Math.Min(au, Math.Min(bu, cu)) + 1f) / cell - 0.5f));
            int u1 = Math.Min(grid - 1, (int)MathF.Ceiling((Math.Max(au, Math.Max(bu, cu)) + 1f) / cell - 0.5f));
            int v0 = Math.Max(0, (int)MathF.Floor((Math.Min(av, Math.Min(bv, cv)) + 1f) / cell - 0.5f));
            int v1 = Math.Min(grid - 1, (int)MathF.Ceiling((Math.Max(av, Math.Max(bv, cv)) + 1f) / cell - 0.5f));
            for (int pu = u0; pu <= u1; pu++)
            {
                float x = (pu + 0.5f) * cell - 1f;
                for (int pv = v0; pv <= v1; pv++)
                {
                    float y = (pv + 0.5f) * cell - 1f;
                    float l1 = ((x - au) * (cv - av) - (cu - au) * (y - av)) / det;
                    float l2 = ((bu - au) * (y - av) - (x - au) * (bv - av)) / det;
                    float l0 = 1f - l1 - l2;
                    const float Eps = -1e-6f;
                    if (l0 < Eps || l1 < Eps || l2 < Eps)
                    {
                        continue;
                    }

                    float depth = l0 * ad + l1 * bd + l2 * cd;
                    int pixel = pu * grid + pv;
                    low[pixel] = Math.Min(low[pixel], depth);
                    high[pixel] = Math.Max(high[pixel], depth);
                }
            }
        }
    }

    private sealed class Graph(int[] offsets, int[] targets, float[] weights)
    {
        public int[] Offsets { get; } = offsets;
        public int[] Targets { get; } = targets;
        public float[] Weights { get; } = weights;
        public int Nodes => Offsets.Length - 1;
    }

    private static Graph BuildGraph(float[] vertices, int[] faces, int n, int m, int grid, float cell, int[] cellNode, float[] centers,
        UniRigVoxelSkinOptions options)
    {
        List<(int A, int B, float W)> edges = new(faces.Length + 4 * m);
        HashSet<long> seen = new(faces.Length);
        for (int f = 0; f < faces.Length / 3; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                int a = faces[3 * f + k];
                int b = faces[3 * f + (k + 1) % 3];
                long key = a < b ? (long)a * n + b : (long)b * n + a;
                if (a != b && seen.Add(key))
                {
                    edges.Add((a, b, Distance(vertices, a, vertices, b)));
                }
            }
        }

        UniRigKdTree vertexTree = new(vertices);
        Span<int> nearIdx = stackalloc int[4];
        Span<float> nearDist = stackalloc float[4];
        for (int v = 0; v < n; v++)
        {
            vertexTree.Nearest(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2], 4, nearIdx, nearDist);
            for (int k = 1; k < 4; k++)
            {
                if (nearIdx[k] >= 0 && nearDist[k] > 0f && nearDist[k] < options.LinkDistance * 1.001f)
                {
                    edges.Add((v, nearIdx[k], nearDist[k]));
                }
            }
        }

        float neighbour = cell * 1.001f;
        for (int c = 0; c < m; c++)
        {
            int node = n + c;
            int x = (int)MathF.Floor((centers[3 * c] + 1f) / cell);
            int y = (int)MathF.Floor((centers[3 * c + 1] + 1f) / cell);
            int z = (int)MathF.Floor((centers[3 * c + 2] + 1f) / cell);
            AddGridEdge(edges, cellNode, grid, node, x + 1, y, z, cell * options.GridWeight);
            AddGridEdge(edges, cellNode, grid, node, x, y + 1, z, cell * options.GridWeight);
            AddGridEdge(edges, cellNode, grid, node, x, y, z + 1, cell * options.GridWeight);
            (int nearest, float distance) = vertexTree.Nearest(centers[3 * c], centers[3 * c + 1], centers[3 * c + 2]);
            if (nearest >= 0 && distance > 0f && distance < neighbour)
            {
                edges.Add((node, nearest, distance));
            }
        }

        int total = n + m;
        int[] offsets = new int[total + 1];
        foreach ((int a, int b, _) in edges)
        {
            offsets[a + 1]++;
            offsets[b + 1]++;
        }

        for (int i = 0; i < total; i++)
        {
            offsets[i + 1] += offsets[i];
        }

        int[] targets = new int[offsets[total]];
        float[] weights = new float[offsets[total]];
        int[] fill = new int[total];
        foreach ((int a, int b, float w) in edges)
        {
            int ia = offsets[a] + fill[a]++;
            targets[ia] = b;
            weights[ia] = w;
            int ib = offsets[b] + fill[b]++;
            targets[ib] = a;
            weights[ib] = w;
        }

        return new Graph(offsets, targets, weights);
    }

    private static void AddGridEdge(List<(int A, int B, float W)> edges, int[] cellNode, int grid, int node, int x, int y, int z, float weight)
    {
        if (x >= grid || y >= grid || z >= grid)
        {
            return;
        }

        int other = cellNode[(x * grid + y) * grid + z];
        if (other >= 0)
        {
            edges.Add((node, other, weight));
        }
    }

    private static int NearestNode(UniRigKdTree vertexTree, float[] centers, int m, int n, float x, float y, float z)
    {
        (int vertex, float vertexDistance) = vertexTree.Nearest(x, y, z);
        float best = vertexDistance;
        int node = vertex;
        for (int c = 0; c < m; c++)
        {
            float dx = centers[3 * c] - x;
            float dy = centers[3 * c + 1] - y;
            float dz = centers[3 * c + 2] - z;
            float d = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
            if (d < best)
            {
                best = d;
                node = n + c;
            }
        }

        return node;
    }

    private static float[] Dijkstra(Graph graph, int source, int vertexCount)
    {
        float[] dist = new float[graph.Nodes];
        Array.Fill(dist, float.PositiveInfinity);
        PriorityQueue<int, float> queue = new();
        dist[source] = 0f;
        queue.Enqueue(source, 0f);
        while (queue.TryDequeue(out int node, out float d))
        {
            if (d > dist[node])
            {
                continue;
            }

            for (int e = graph.Offsets[node]; e < graph.Offsets[node + 1]; e++)
            {
                int target = graph.Targets[e];
                float next = d + graph.Weights[e];
                if (next < dist[target])
                {
                    dist[target] = next;
                    queue.Enqueue(target, next);
                }
            }
        }

        return dist[..vertexCount];
    }

    private static float[] Weights(float[] vertices, float[] joints, float[][] distance, int n, int j, float alpha)
    {
        // A vertex no joint reaches takes the Euclidean distance of its three nearest joints, as upstream does.
        UniRigKdTree jointTree = new(joints);
        int k = Math.Min(j, 3);
        Span<int> idx = stackalloc int[3];
        Span<float> dis = stackalloc float[3];
        float max = 0f;
        for (int joint = 0; joint < j; joint++)
        {
            foreach (float d in distance[joint])
            {
                if (float.IsFinite(d))
                {
                    max = Math.Max(max, d);
                }
            }
        }

        for (int v = 0; v < n; v++)
        {
            bool reached = false;
            for (int joint = 0; joint < j && !reached; joint++)
            {
                reached = float.IsFinite(distance[joint][v]);
            }

            if (reached)
            {
                continue;
            }

            jointTree.Nearest(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2], k, idx, dis);
            for (int q = 0; q < k; q++)
            {
                distance[idx[q]][v] = dis[q];
                max = Math.Max(max, dis[q]);
            }
        }

        float[] skin = new float[n * j];
        for (int v = 0; v < n; v++)
        {
            double sum = 0;
            for (int joint = 0; joint < j; joint++)
            {
                float d = float.IsFinite(distance[joint][v]) ? distance[joint][v] : max;
                d = Math.Max(d, 1e-6f);
                double denominator = (1 - alpha) * d + alpha * d * d;
                double w = 1.0 / (denominator * denominator);
                skin[v * j + joint] = (float)w;
                sum += w;
            }

            for (int joint = 0; joint < j; joint++)
            {
                skin[v * j + joint] = (float)(skin[v * j + joint] / sum);
            }
        }

        return skin;
    }

    private static float Distance(float[] a, int ai, float[] b, int bi)
    {
        float dx = a[3 * ai] - b[3 * bi];
        float dy = a[3 * ai + 1] - b[3 * bi + 1];
        float dz = a[3 * ai + 2] - b[3 * bi + 2];
        return MathF.Sqrt(dx * dx + dy * dy + dz * dz);
    }
}
