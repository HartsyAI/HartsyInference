namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Traces the boundary of a triangle mesh into loops. A boundary edge has one face; at a vertex where several boundary loops touch (a pinch)
/// each loop is the boundary of one fan of faces, so loops that only share a vertex stay separate instead of merging into one figure-eight.</summary>
public static class MeshBoundary
{
    /// <summary>Returns every boundary loop of <paramref name="mesh"/> by vertex index, in the order of the lowest half-edge slot of each. A chain that meets an edge
    /// with more than two faces ends there and comes out whole, from its head to its end, whichever face order the mesh has. Faces that repeat a vertex are ignored.</summary>
    public static IReadOnlyList<BoundaryLoop> Trace(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        int vertexCount = mesh.VertexCount;
        int[] indices = mesh.Indices;
        bool[]? alive = LiveFaces(indices);
        EdgeTable table = EdgeTable.Build(indices, vertexCount, alive);
        List<int> starts = [];
        for (int s = 0; s < table.Count;)
        {
            int e = table.RunEnd(s);
            if (e - s == 1) starts.Add(table.HalfEdges[s]);
            s = e;
        }
        if (starts.Count == 0) return [];
        starts.Sort();

        (int[] first, int[] faces) = VertexFaces(indices, vertexCount, alive);
        bool[] visited = new bool[indices.Length];
        List<BoundaryLoop> loops = [];
        List<int> vertices = [];
        foreach (int origin in starts)
        {
            if (visited[origin]) continue;
            (int start, int u, int v, int face) = Head(indices, first, faces, origin, starts.Count);
            visited[start] = true;
            vertices.Clear();
            vertices.Add(u);
            bool closed;
            while (true)
            {
                if (!TryNextEdge(indices, first, faces, face, u, v, out int nextFace, out int nextTo, out int slot))
                {
                    vertices.Add(v);
                    closed = false;
                    break;
                }
                if (slot == start)
                {
                    closed = true;
                    break;
                }
                if (visited[slot])
                {
                    vertices.Add(v);
                    closed = false;
                    break;
                }
                visited[slot] = true;
                vertices.Add(v);
                u = v;
                v = nextTo;
                face = nextFace;
            }
            loops.Add(new BoundaryLoop([.. vertices], closed, Length(mesh.Vertices, vertices, closed)));
        }
        return loops;
    }

    // Walks backwards from the edge in slot `origin` to the first edge of its chain; a closed loop gives `origin` itself.
    private static (int Slot, int From, int To, int Face) Head(int[] indices, int[] first, int[] faces, int origin, int limit)
    {
        int slot = origin, from = indices[origin], to = indices[NextCorner(origin)], face = origin / 3;
        for (int step = 0; step <= limit; step++)
        {
            if (!TryNextEdge(indices, first, faces, face, to, from, out int previousFace, out int previousFrom, out int previousSlot)
                || previousSlot == origin)
                break;
            to = from;
            from = previousFrom;
            face = previousFace;
            slot = previousSlot;
        }
        return slot == origin ? (origin, indices[origin], indices[NextCorner(origin)], origin / 3) : (slot, from, to, face);
    }

    // From the boundary edge (u, v) of `face`, turns around v through the faces of its fan until the next boundary edge.
    private static bool TryNextEdge(
        int[] indices, int[] first, int[] faces, int face, int u, int v, out int nextFace, out int nextTo, out int slot)
    {
        int limit = first[v + 1] - first[v] + 1;
        for (int step = 0; step < limit; step++)
        {
            int c = Third(indices, face, u, v);
            int other = -1, count = 0;
            for (int k = first[v]; k < first[v + 1]; k++)
            {
                int g = faces[k];
                if (g == face || !Has(indices, g, c)) continue;
                other = g;
                count++;
            }
            if (count == 0)
            {
                nextFace = face;
                nextTo = c;
                slot = SlotOf(indices, face, v, c);
                return true;
            }
            if (count > 1) break;
            face = other;
            u = c;
        }
        nextFace = nextTo = slot = -1;
        return false;
    }

    // Null when every face has three distinct vertices.
    private static bool[]? LiveFaces(int[] indices)
    {
        bool[]? alive = null;
        for (int f = 0; f < indices.Length / 3; f++)
        {
            int a = indices[f * 3], b = indices[f * 3 + 1], c = indices[f * 3 + 2];
            if (a != b && b != c && a != c) continue;
            alive ??= Enumerable.Repeat(true, indices.Length / 3).ToArray();
            alive[f] = false;
        }
        return alive;
    }

    private static (int[] First, int[] Faces) VertexFaces(int[] indices, int vertexCount, bool[]? alive)
    {
        int[] first = new int[vertexCount + 1];
        for (int c = 0; c < indices.Length; c++)
            if (alive is null || alive[c / 3]) first[indices[c] + 1]++;
        for (int i = 0; i < vertexCount; i++) first[i + 1] += first[i];
        int[] faces = new int[first[vertexCount]];
        int[] fill = new int[vertexCount];
        for (int c = 0; c < indices.Length; c++)
        {
            if (alive is not null && !alive[c / 3]) continue;
            int v = indices[c];
            faces[first[v] + fill[v]++] = c / 3;
        }
        return (first, faces);
    }

    private static int Third(int[] indices, int face, int a, int b)
    {
        for (int c = 0; c < 3; c++)
        {
            int v = indices[face * 3 + c];
            if (v != a && v != b) return v;
        }
        return -1;
    }

    private static bool Has(int[] indices, int face, int v) =>
        indices[face * 3] == v || indices[face * 3 + 1] == v || indices[face * 3 + 2] == v;

    private static int SlotOf(int[] indices, int face, int a, int b)
    {
        for (int c = 0; c < 3; c++)
        {
            int from = indices[face * 3 + c], to = indices[face * 3 + (c + 1) % 3];
            if ((from == a && to == b) || (from == b && to == a)) return face * 3 + c;
        }
        return -1;
    }

    private static int NextCorner(int slot) => slot / 3 * 3 + (slot % 3 + 1) % 3;

    private static double Length(float[] v, List<int> vertices, bool closed)
    {
        double sum = 0;
        int edges = closed ? vertices.Count : vertices.Count - 1;
        for (int i = 0; i < edges; i++)
        {
            int a = vertices[i] * 3, b = vertices[(i + 1) % vertices.Count] * 3;
            double dx = v[a] - v[b], dy = v[a + 1] - v[b + 1], dz = v[a + 2] - v[b + 2];
            sum += Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
        return sum;
    }
}
