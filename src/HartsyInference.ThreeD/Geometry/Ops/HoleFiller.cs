namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>What <see cref="HoleFiller.Fill"/> did. Every simple boundary cycle of three or more vertices ends up filled, too wide, or one the triangulation could not close; the last two are what is still open.</summary>
/// <param name="Mesh">The mesh with the fills added (the input itself when nothing was added).</param>
/// <param name="Filled">Cycles closed.</param>
/// <param name="TooWide">Cycles left open because their triangulation needs an edge longer than the limit.</param>
/// <param name="Unclosable">Cycles left open for another reason: a triangle that already exists, or no ear that adds no existing edge.</param>
/// <param name="OpenChains">Boundary walks that end at a non-manifold edge; they are not filled.</param>
/// <param name="TrianglesAdded">Triangles added.</param>
public readonly record struct HoleFillResult(Mesh Mesh, int Filled, int TooWide, int Unclosable, int OpenChains, int TrianglesAdded);

/// <summary>Closes boundary loops by ear clipping, best-shaped ear first, so a long thin slit becomes a zipper of triangles across its width and a round hole a spiral of ears.
/// A loop is closed only when every new edge stays within a width limit; that is what keeps a wide opening (a cut base, a mouth) open while a slit of any length closes.
/// Loops that touch at a pinch vertex are closed lobe by lobe (<see cref="BoundaryLoop.SimpleCycles"/>). No vertex is added.</summary>
public static class HoleFiller
{
    private const double ReflexPenalty = 10;

    // A longer limit lets an early ear take a long edge and strand the rest of the loop, so a loop is tried at 1/16, 1/8, 1/4, 1/2 and the full limit, narrowest first,
    // and a tier that fails for any reason hands over to the next (the ear order depends on the limit).
    private const int WidthTiers = 5;

    /// <summary>Closes every boundary loop of <paramref name="mesh"/> whose triangulation needs no new edge longer than <paramref name="maxWidth"/> (mesh units). The patches wind opposite to the loop's faces, so a
    /// consistently wound mesh stays consistent. A pass can free loops the one before refused (its new edges exist now), so passes repeat up to <paramref name="maxPasses"/> while they add triangles;
    /// the open counts are those of the last pass, taken before it filled. The result shares <paramref name="mesh"/>'s vertex array.</summary>
    public static HoleFillResult Fill(Mesh mesh, double maxWidth, int maxPasses = 4)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (!(maxWidth >= 0))
            throw new ArgumentOutOfRangeException(nameof(maxWidth), "Width must be non-negative.");
        ArgumentOutOfRangeException.ThrowIfLessThan(maxPasses, 1);
        HoleFillResult total = Pass(mesh, maxWidth);
        for (int pass = 1; pass < maxPasses && total.TrianglesAdded > 0; pass++)
        {
            HoleFillResult next = Pass(total.Mesh, maxWidth);
            if (next.TrianglesAdded == 0)
            {
                total = total with { TooWide = next.TooWide, Unclosable = next.Unclosable, OpenChains = next.OpenChains };
                break;
            }
            total = next with { Filled = total.Filled + next.Filled, TrianglesAdded = total.TrianglesAdded + next.TrianglesAdded };
        }
        return total;
    }

    private static HoleFillResult Pass(Mesh mesh, double maxWidth)
    {
        IReadOnlyList<BoundaryLoop> walks = MeshBoundary.Trace(mesh);
        if (walks.Count == 0) return new HoleFillResult(mesh, 0, 0, 0, 0, 0);

        int vertexCount = mesh.VertexCount;
        EdgeTable table = EdgeTable.Build(mesh.Indices, vertexCount);
        HashSet<long> added = [];
        HashSet<(int, int, int)>? faces = null;
        List<int> triangles = [];
        List<int> patch = [];
        HashSet<long> patchEdges = [];
        int filled = 0, tooWide = 0, unclosable = 0, open = 0;
        foreach (BoundaryLoop walk in walks)
        {
            if (!walk.Closed)
            {
                open++;
                continue;
            }
            foreach (int[] cycle in walk.SimpleCycles())
            {
                Outcome outcome = Outcome.Unclosable;
                Vector3d[] normal = SurfaceNormals(mesh, table, cycle);
                int tiers = maxWidth > 0 ? WidthTiers : 1;
                for (int tier = 0; tier < tiers && outcome != Outcome.Closed; tier++)
                {
                    patch.Clear();
                    patchEdges.Clear();
                    outcome = Triangulate(mesh, table, cycle, normal, added, maxWidth / (1 << (tiers - 1 - tier)), patch, patchEdges);
                }
                if (outcome == Outcome.Closed && cycle.Length == 3
                    && (faces ??= FaceSet(mesh.Indices)).Contains(Sorted(cycle[0], cycle[1], cycle[2])))
                    outcome = Outcome.Unclosable;
                switch (outcome)
                {
                    case Outcome.Closed:
                        filled++;
                        triangles.AddRange(patch);
                        foreach (long key in patchEdges) added.Add(key);
                        break;
                    case Outcome.TooWide:
                        tooWide++;
                        break;
                    default:
                        unclosable++;
                        break;
                }
            }
        }
        if (triangles.Count == 0) return new HoleFillResult(mesh, 0, tooWide, unclosable, open, 0);
        Mesh result = new() { Vertices = mesh.Vertices, Indices = [.. mesh.Indices, .. triangles] };
        return new HoleFillResult(result, filled, tooWide, unclosable, open, triangles.Count / 3);
    }

    private enum Outcome
    {
        Closed,
        TooWide,
        Unclosable,
    }

    // Ear clipping on the loop. An ear is ranked by how round its triangle is (a sliver along one side of a slit comes last, so the clipping zips across the slit),
    // and an ear that folds against the surrounding surface comes after every other. An ear whose new edge already exists in the mesh or in an earlier patch is skipped,
    // since a face on it would give that edge a third face; one whose new edge is over the width limit is skipped until its neighbours change.
    private static Outcome Triangulate(
        Mesh mesh, EdgeTable table, int[] loop, Vector3d[] normal, HashSet<long> added, double maxWidth, List<int> triangles, HashSet<long> newEdges)
    {
        float[] v = mesh.Vertices;
        int vertexCount = mesh.VertexCount;
        int n = loop.Length;
        int[] prev = new int[n], next = new int[n], stamp = new int[n];
        for (int i = 0; i < n; i++)
        {
            prev[i] = (i + n - 1) % n;
            next[i] = (i + 1) % n;
        }
        PriorityQueue<(int Ear, int Stamp), double> queue = new();
        for (int i = 0; i < n; i++) Enqueue(i);
        bool[] gone = new bool[n];
        int alive = n;
        bool wide = false;
        // A triangle hole adds no edge, so its own edges are the width.
        if (n == 3 && Math.Max(Distance(v, loop[0], loop[1]), Math.Max(Distance(v, loop[1], loop[2]), Distance(v, loop[2], loop[0]))) > maxWidth)
            return Outcome.TooWide;
        while (alive > 3)
        {
            if (!queue.TryDequeue(out (int Ear, int Stamp) top, out _)) return wide ? Outcome.TooWide : Outcome.Unclosable;
            int ear = top.Ear;
            if (gone[ear] || stamp[ear] != top.Stamp) continue;
            int a = prev[ear], c = next[ear];
            if (Distance(v, loop[a], loop[c]) > maxWidth)
            {
                wide = true;
                continue;
            }
            triangles.AddRange([loop[a], loop[c], loop[ear]]);
            newEdges.Add(EdgeTable.Key(loop[a], loop[c], vertexCount));
            gone[ear] = true;
            alive--;
            next[a] = c;
            prev[c] = a;
            stamp[a]++;
            stamp[c]++;
            Enqueue(a);
            Enqueue(c);
        }
        int last = Array.FindIndex(gone, g => !g);
        triangles.AddRange([loop[prev[last]], loop[next[last]], loop[last]]);
        return Outcome.Closed;

        void Enqueue(int i)
        {
            int a = prev[i], c = next[i];
            long key = EdgeTable.Key(loop[a], loop[c], vertexCount);
            if (Array.BinarySearch(table.Keys, key) >= 0 || added.Contains(key) || newEdges.Contains(key)) return;
            Vector3d ab = Sub(v, loop[i], loop[a]), bc = Sub(v, loop[c], loop[i]), ac = Sub(v, loop[c], loop[a]);
            Vector3d cross = Vector3d.Cross(ac, ab);
            double squares = Vector3d.Dot(ab, ab) + Vector3d.Dot(bc, bc) + Vector3d.Dot(ac, ac);
            double roundness = squares > 0 ? 2 * Math.Sqrt(3) * cross.Length / squares : 0;
            bool folds = Vector3d.Dot(cross, normal[a] + normal[i] + normal[c]) < 0;
            double reach = maxWidth > 0 ? Math.Min(1, Math.Sqrt(Vector3d.Dot(ac, ac)) / maxWidth) : 0;
            queue.Enqueue((i, stamp[i]), (folds ? ReflexPenalty : 0) + 1 - roundness + reach);
        }
    }

    // Outward direction of the surface at each loop vertex: the normals of the faces on the two loop edges that meet there (the loop walks along the faces' own winding).
    private static Vector3d[] SurfaceNormals(Mesh mesh, EdgeTable table, int[] loop)
    {
        int n = loop.Length;
        Vector3d[] edgeNormal = new Vector3d[n];
        for (int k = 0; k < n; k++)
        {
            int at = Array.BinarySearch(table.Keys, EdgeTable.Key(loop[k], loop[(k + 1) % n], mesh.VertexCount));
            if (at < 0) continue;
            int face = table.HalfEdges[at] / 3;
            int a = mesh.Indices[face * 3], b = mesh.Indices[face * 3 + 1], c = mesh.Indices[face * 3 + 2];
            edgeNormal[k] = Vector3d.Cross(Sub(mesh.Vertices, b, a), Sub(mesh.Vertices, c, a));
        }
        Vector3d[] at2 = new Vector3d[n];
        for (int k = 0; k < n; k++) at2[k] = edgeNormal[k] + edgeNormal[(k + n - 1) % n];
        return at2;
    }

    private static double Distance(float[] v, int a, int b) => Sub(v, a, b).Length;

    private static Vector3d Sub(float[] v, int a, int b) =>
        new((double)v[a * 3] - v[b * 3], (double)v[a * 3 + 1] - v[b * 3 + 1], (double)v[a * 3 + 2] - v[b * 3 + 2]);

    private static (int, int, int) Sorted(int a, int b, int c)
    {
        int lo = Math.Min(a, Math.Min(b, c)), hi = Math.Max(a, Math.Max(b, c));
        return (lo, a + b + c - lo - hi, hi);
    }

    private static HashSet<(int, int, int)> FaceSet(int[] indices)
    {
        HashSet<(int, int, int)> set = new(indices.Length / 3);
        for (int f = 0; f < indices.Length / 3; f++)
            set.Add(Sorted(indices[f * 3], indices[f * 3 + 1], indices[f * 3 + 2]));
        return set;
    }
}
