namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Topology repair for indexed triangle meshes: vertex welding, degenerate and duplicate face removal, non-manifold edge splitting,
/// small-component removal, hole filling and orientation unification. Every step returns a new geometry-only mesh (positions and
/// indices; normals, UVs and colors are dropped) with unreferenced vertices removed and the surviving order preserved.
/// </summary>
public static class MeshCleaner
{
    /// <summary>Runs the steps enabled in <paramref name="options"/> in the order: weld, degenerate, duplicate, small components by face count, non-manifold, small
    /// components by area, holes (perimeter, then width), orientation.</summary>
    public static Mesh Clean(Mesh mesh, MeshCleanOptions? options = null)
    {
        Validate(mesh);
        options ??= new MeshCleanOptions();
        Mesh m = Compact(mesh.Vertices, mesh.Indices);
        if (options.WeldVertices) m = WeldVertices(m, options.WeldTolerance);
        if (options.RemoveDegenerateFaces) m = RemoveDegenerateFaces(m);
        if (options.RemoveDuplicateFaces) m = RemoveDuplicateFaces(m);
        // The face-count rule goes before the non-manifold split, which cuts vertices and so could separate faces that MeshLab sees as one component.
        if (options.SmallComponentFaceRatio > 0) m = RemoveSmallComponentsByFaceCount(m, options.SmallComponentFaceRatio);
        if (options.RepairNonManifoldEdges) m = RepairNonManifoldEdges(m);
        if (options.SmallComponentAreaRatio > 0) m = RemoveSmallComponents(m, options.SmallComponentAreaRatio);
        if (options.HoleFillMaxPerimeter > 0) m = FillHoles(m, options.HoleFillMaxPerimeter);
        if (options.HoleFillMaxWidth > 0)
        {
            // The patches take their winding from the faces around the hole, so those must agree first.
            m = HoleFiller.Fill(UnifyOrientation(m, orientOutward: false), options.HoleFillMaxWidth).Mesh;
        }
        if (options.UnifyOrientation) m = UnifyOrientation(m, options.OrientOutward);
        return m;
    }

    /// <summary>Merges vertices closer than <paramref name="tolerance"/> (0: identical positions only) by snapping to a grid of that size.</summary>
    public static Mesh WeldVertices(Mesh mesh, double tolerance = 0)
    {
        Validate(mesh);
        if (tolerance < 0 || double.IsNaN(tolerance))
            throw new ArgumentOutOfRangeException(nameof(tolerance), "Tolerance must be non-negative.");
        float[] v = mesh.Vertices;
        int count = mesh.VertexCount;
        int[] remap = new int[count];
        List<float> kept = new(v.Length);
        if (tolerance == 0)
        {
            Dictionary<(float, float, float), int> seen = new(count);
            for (int i = 0; i < count; i++)
            {
                (float, float, float) key = (v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
                if (!seen.TryGetValue(key, out int id))
                {
                    id = kept.Count / 3;
                    seen[key] = id;
                    kept.Add(key.Item1); kept.Add(key.Item2); kept.Add(key.Item3);
                }
                remap[i] = id;
            }
        }
        else
        {
            Dictionary<(long, long, long), int> seen = new(count);
            for (int i = 0; i < count; i++)
            {
                (long, long, long) key = ((long)Math.Floor(v[i * 3] / tolerance), (long)Math.Floor(v[i * 3 + 1] / tolerance),
                    (long)Math.Floor(v[i * 3 + 2] / tolerance));
                if (!seen.TryGetValue(key, out int id))
                {
                    id = kept.Count / 3;
                    seen[key] = id;
                    kept.Add(v[i * 3]); kept.Add(v[i * 3 + 1]); kept.Add(v[i * 3 + 2]);
                }
                remap[i] = id;
            }
        }
        int[] indices = new int[mesh.Indices.Length];
        for (int i = 0; i < indices.Length; i++)
            indices[i] = remap[mesh.Indices[i]];
        return Compact([.. kept], indices);
    }

    /// <summary>Removes faces that repeat a vertex or whose area is zero.</summary>
    public static Mesh RemoveDegenerateFaces(Mesh mesh)
    {
        Validate(mesh);
        int faces = mesh.TriangleCount;
        bool[] keep = new bool[faces];
        for (int f = 0; f < faces; f++)
        {
            int a = mesh.Indices[f * 3], b = mesh.Indices[f * 3 + 1], c = mesh.Indices[f * 3 + 2];
            keep[f] = a != b && b != c && a != c && FaceArea(mesh.Vertices, a, b, c) > 1e-30;
        }
        return Rebuild(mesh, keep);
    }

    /// <summary>Removes faces that use the same three vertices as an earlier face, in either winding.</summary>
    public static Mesh RemoveDuplicateFaces(Mesh mesh)
    {
        Validate(mesh);
        int faces = mesh.TriangleCount;
        bool[] keep = new bool[faces];
        HashSet<(int, int, int)> seen = new(faces);
        for (int f = 0; f < faces; f++)
        {
            int a = mesh.Indices[f * 3], b = mesh.Indices[f * 3 + 1], c = mesh.Indices[f * 3 + 2];
            int lo = Math.Min(a, Math.Min(b, c)), hi = Math.Max(a, Math.Max(b, c));
            keep[f] = seen.Add((lo, a + b + c - lo - hi, hi));
        }
        return Rebuild(mesh, keep);
    }

    /// <summary>Makes every edge belong to at most two faces by splitting vertices, keeping every face: upstream's <c>repair_non_manifold_edges</c>
    /// ("repair non-manifold edges by splitting vertices", duplicates with the same coordinates). For an edge shared by more than two faces the
    /// consistently wound, largest pair stays joined; the other faces are cut off along it, which gives them copies of its two vertices. Corners
    /// of one vertex stay one vertex while a chain of joined edges links their faces. Joins made for the next edge of a chain can link the
    /// faces of an edge whose own pair was cut off again at both ends; such an edge is then joined nowhere (its kept pair included) and the
    /// grouping is redone. Removing joins can only split groups further, so one redo is enough: the second pass finds nothing new. No edge keeps
    /// more than two faces afterwards: a corner has two edges at its vertex and each gives it at most one join, so the corners of a vertex
    /// group into paths and cycles; the faces of a cut edge are path ends, at most two per group. That assumes no face repeats a vertex
    /// (<see cref="RemoveDegenerateFaces"/> runs first in <see cref="Clean"/>). Positions and faces are unchanged, so nothing is removed from the
    /// surface; <see cref="RemoveNonManifoldFaces"/> is the variant that drops faces.</summary>
    public static Mesh RepairNonManifoldEdges(Mesh mesh)
    {
        Validate(mesh);
        int faces = mesh.TriangleCount;
        int vertexCount = mesh.VertexCount;
        EdgeTable table = EdgeTable.Build(mesh.Indices, vertexCount);

        // The edges with more than two faces, their runs in the table, and the pair of half-edges each keeps joined.
        List<(int Start, int End)> overfull = [];
        List<(int First, int Second)> kept = [];
        List<int> run = [];
        for (int s = 0; s < table.Count;)
        {
            int e = table.RunEnd(s);
            if (e - s > 2)
            {
                run.Clear();
                for (int k = s; k < e; k++) run.Add(table.HalfEdges[k]);
                (int bestI, int bestJ) = BestPair(mesh, table, run);
                overfull.Add((s, e));
                kept.Add((run[bestI], run[bestJ]));
            }
            s = e;
        }
        if (overfull.Count == 0) return Compact(mesh.Vertices, mesh.Indices);

        bool[] cut = new bool[overfull.Count];
        int[] parent = new int[faces * 3];
        while (true)
        {
            for (int i = 0; i < parent.Length; i++) parent[i] = i;
            for (int s = 0; s < table.Count;)
            {
                int e = table.RunEnd(s);
                if (e - s == 2) JoinAcross(table, parent, table.HalfEdges[s], table.HalfEdges[s + 1]);
                s = e;
            }
            for (int k = 0; k < overfull.Count; k++)
                if (!cut[k]) JoinAcross(table, parent, kept[k].First, kept[k].Second);

            (int[] indices, List<int> sources) = SplitCornerGroups(mesh.Indices, parent, vertexCount);
            bool changed = false;
            Dictionary<long, int> shared = [];
            for (int k = 0; k < overfull.Count; k++)
            {
                if (cut[k]) continue;
                shared.Clear();
                for (int h = overfull[k].Start; h < overfull[k].End; h++)
                {
                    int slot = table.HalfEdges[h];
                    long key = EdgeTable.Key(indices[slot], indices[NextCorner(slot)], vertexCount + sources.Count);
                    int count = shared.GetValueOrDefault(key) + 1;
                    shared[key] = count;
                    if (count > 2) cut[k] = true;
                }
                changed |= cut[k];
            }
            if (changed) continue;

            float[] vertices = new float[(vertexCount + sources.Count) * 3];
            Array.Copy(mesh.Vertices, vertices, mesh.Vertices.Length);
            for (int i = 0; i < sources.Count; i++)
                Array.Copy(mesh.Vertices, sources[i] * 3, vertices, (vertexCount + i) * 3, 3);
            return Compact(vertices, indices);
        }
    }

    // A corner group is one vertex: the group met first keeps the original index, every other group gets a copy (sources[i] is the original
    // vertex of copy vertexCount + i).
    private static (int[] Indices, List<int> Sources) SplitCornerGroups(int[] original, int[] parent, int vertexCount)
    {
        int[] firstRoot = new int[vertexCount];
        Array.Fill(firstRoot, -1);
        Dictionary<int, int> copyOf = [];
        List<int> sources = [];
        int[] indices = new int[original.Length];
        for (int c = 0; c < indices.Length; c++)
        {
            int v = original[c];
            int root = Find(parent, c);
            if (firstRoot[v] < 0) firstRoot[v] = root;
            if (firstRoot[v] == root)
            {
                indices[c] = v;
                continue;
            }
            if (!copyOf.TryGetValue(root, out int id))
            {
                id = vertexCount + sources.Count;
                copyOf[root] = id;
                sources.Add(v);
            }
            indices[c] = id;
        }
        return (indices, sources);
    }

    // Joins the corners of two faces across their shared edge: slot h is the corner at the edge's first vertex, the next corner the second.
    private static void JoinAcross(EdgeTable table, int[] parent, int h1, int h2)
    {
        int n1 = NextCorner(h1), n2 = NextCorner(h2);
        if (table.From(h1) == table.From(h2))
        {
            Union(parent, h1, h2);
            Union(parent, n1, n2);
        }
        else
        {
            Union(parent, h1, n2);
            Union(parent, n1, h2);
        }
    }

    private static int NextCorner(int slot) => slot / 3 * 3 + (slot % 3 + 1) % 3;

    /// <summary>For every edge shared by more than two faces keeps the pair that is consistently wound and largest, and drops the rest
    /// (a crack opens along the edge; <see cref="RepairNonManifoldEdges"/> splits vertices instead).</summary>
    public static Mesh RemoveNonManifoldFaces(Mesh mesh)
    {
        Validate(mesh);
        int faces = mesh.TriangleCount;
        EdgeTable table = EdgeTable.Build(mesh.Indices, mesh.VertexCount);
        bool[] keep = new bool[faces];
        Array.Fill(keep, true);
        List<int> run = [];
        for (int s = 0; s < table.Count;)
        {
            int e = table.RunEnd(s);
            if (e - s > 2)
            {
                run.Clear();
                for (int k = s; k < e; k++)
                    if (keep[table.HalfEdges[k] / 3]) run.Add(table.HalfEdges[k]);
                if (run.Count > 2) ThinRun(mesh, table, run, keep);
            }
            s = e;
        }
        return Rebuild(mesh, keep);
    }

    /// <summary>Removes the edge-connected components (faces linked through shared edges, as MeshLab's face-face adjacency links them; faces that only touch at a vertex are separate) that fall under the
    /// threshold of MeshLab's <c>SmallComponent::Select</c> (<c>smallcomponentselection.h</c>, read 2026-11-13), the rule of upstream's <c>remove_floater</c> (pymeshlab <c>nbfaceratio</c> 0.005): the
    /// threshold is <c>(uint)((float)largest * (float)faceRatio)</c>, a single-precision product truncated to an integer, and a component stays when its face count is at least that. With a largest of 1,000
    /// faces and a ratio of 0.005 a 5-face piece stays and a 4-face piece goes; with 1,100 the threshold is 5.5 truncated to 5, so a 5-face piece still stays.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="faceRatio"/> is negative, above 1 (the rule would remove the largest component too) or not a number.</exception>
    public static Mesh RemoveSmallComponentsByFaceCount(Mesh mesh, double faceRatio) => Rebuild(mesh, FaceCountComponentMask(mesh, faceRatio));

    /// <summary>For each face, whether its edge-connected component stays under <see cref="RemoveSmallComponentsByFaceCount"/>'s rule.</summary>
    internal static bool[] FaceCountComponentMask(Mesh mesh, double faceRatio)
    {
        Validate(mesh);
        if (!(faceRatio >= 0 && faceRatio <= 1))
            throw new ArgumentOutOfRangeException(nameof(faceRatio), "Ratio must be between 0 and 1.");
        int faces = mesh.TriangleCount;
        int[] parent = new int[faces];
        for (int i = 0; i < faces; i++) parent[i] = i;
        Dictionary<long, int> firstFace = new(faces * 3 / 2 + 1);
        for (int f = 0; f < faces; f++)
        {
            for (int e = 0; e < 3; e++)
            {
                int a = mesh.Indices[f * 3 + e], b = mesh.Indices[f * 3 + (e + 1) % 3];
                long key = a < b ? ((long)a << 32) | (uint)b : ((long)b << 32) | (uint)a;
                if (firstFace.TryGetValue(key, out int other))
                    Union(parent, f, other);
                else
                    firstFace[key] = f;
            }
        }

        int[] size = new int[faces];
        for (int f = 0; f < faces; f++)
            size[Find(parent, f)]++;
        int largest = size.Length == 0 ? 0 : size.Max();
        float product = (float)largest * (float)faceRatio; // single precision and truncated, as in MeshLab: uint th = max(largest, remaining) * nbFaceRatio
        uint threshold = (uint)product;
        bool[] keep = new bool[faces];
        for (int f = 0; f < faces; f++)
            keep[f] = size[Find(parent, f)] >= threshold;
        return keep;
    }

    /// <summary>Removes vertex-connected components below <paramref name="areaRatio"/> of the total area; the largest always stays.</summary>
    public static Mesh RemoveSmallComponents(Mesh mesh, double areaRatio) => Rebuild(mesh, SmallComponentFaceMask(mesh, areaRatio));

    /// <summary>For each face of <paramref name="mesh"/>, whether it belongs to a vertex-connected component that stays under <see cref="RemoveSmallComponents"/>'s rule (the largest always stays, any other needs <paramref name="areaRatio"/> of the total area). Exposed so a caller that has to keep per-vertex attributes can apply the same rule to a welded copy and cut its own mesh.</summary>
    internal static bool[] SmallComponentFaceMask(Mesh mesh, double areaRatio)
    {
        Validate(mesh);
        if (areaRatio < 0 || double.IsNaN(areaRatio))
            throw new ArgumentOutOfRangeException(nameof(areaRatio), "Ratio must be non-negative.");
        int faces = mesh.TriangleCount;
        int[] parent = new int[mesh.VertexCount];
        for (int i = 0; i < parent.Length; i++) parent[i] = i;
        for (int f = 0; f < faces; f++)
        {
            Union(parent, mesh.Indices[f * 3], mesh.Indices[f * 3 + 1]);
            Union(parent, mesh.Indices[f * 3], mesh.Indices[f * 3 + 2]);
        }
        double[] area = new double[mesh.VertexCount];
        double total = 0, largest = 0;
        for (int f = 0; f < faces; f++)
        {
            double a = FaceArea(mesh.Vertices, mesh.Indices[f * 3], mesh.Indices[f * 3 + 1], mesh.Indices[f * 3 + 2]);
            area[Find(parent, mesh.Indices[f * 3])] += a;
            total += a;
        }
        foreach (double a in area)
            largest = Math.Max(largest, a);
        bool[] keep = new bool[faces];
        for (int f = 0; f < faces; f++)
        {
            double a = area[Find(parent, mesh.Indices[f * 3])];
            keep[f] = a >= largest || a >= areaRatio * total;
        }
        return keep;
    }

    /// <summary>Closes boundary loops whose perimeter is at most <paramref name="maxPerimeter"/> (mesh units): a triangle for three
    /// vertices, otherwise a fan around a new centroid vertex. Loops that cannot be traced to a closed cycle are left open.</summary>
    public static Mesh FillHoles(Mesh mesh, double maxPerimeter)
    {
        Validate(mesh);
        if (!(maxPerimeter >= 0))
            throw new ArgumentOutOfRangeException(nameof(maxPerimeter), "Perimeter must be non-negative.");
        EdgeTable table = EdgeTable.Build(mesh.Indices, mesh.VertexCount);
        List<(int From, int To)> boundary = [];
        for (int s = 0; s < table.Count;)
        {
            int e = table.RunEnd(s);
            if (e - s == 1)
                boundary.Add((table.From(table.HalfEdges[s]), table.To(table.HalfEdges[s])));
            s = e;
        }
        if (boundary.Count == 0)
            return Compact(mesh.Vertices, mesh.Indices);
        boundary.Sort();

        Dictionary<int, List<int>> outgoing = [];
        for (int i = 0; i < boundary.Count; i++)
        {
            if (!outgoing.TryGetValue(boundary[i].From, out List<int>? list))
                outgoing[boundary[i].From] = list = [];
            list.Add(i);
        }

        List<float> verts = [.. mesh.Vertices];
        List<int> idx = [.. mesh.Indices];
        bool[] used = new bool[boundary.Count];
        List<int> loop = [];
        HashSet<(int, int, int)>? faceSet = null;
        for (int start = 0; start < boundary.Count; start++)
        {
            if (used[start]) continue;
            loop.Clear();
            used[start] = true;
            loop.Add(boundary[start].From);
            int cur = start;
            bool closed = false;
            while (true)
            {
                int next = boundary[cur].To;
                if (next == loop[0]) { closed = true; break; }
                loop.Add(next);
                int following = -1;
                if (outgoing.TryGetValue(next, out List<int>? options))
                    foreach (int o in options)
                        if (!used[o]) { following = o; break; }
                if (following < 0) break;
                used[following] = true;
                cur = following;
            }
            if (!closed || loop.Count < 3 || LoopPerimeter(mesh.Vertices, loop) > maxPerimeter)
                continue;
            // A lone triangle (a fin split off by the non-manifold repair) is a three-edge loop whose cap would be the same face reversed.
            if (loop.Count == 3 && (faceSet ??= FaceSet(mesh.Indices)).Contains(SortedTriple(loop[0], loop[1], loop[2])))
                continue;
            AddFill(verts, idx, loop);
        }
        return Compact([.. verts], [.. idx]);
    }

    /// <summary>Makes neighboring faces agree on winding by flood fill across manifold edges; with <paramref name="orientOutward"/> closed
    /// components with negative signed volume are then flipped as a whole.</summary>
    public static Mesh UnifyOrientation(Mesh mesh, bool orientOutward = true)
    {
        Validate(mesh);
        int faces = mesh.TriangleCount;
        int[] indices = (int[])mesh.Indices.Clone();
        EdgeTable table = EdgeTable.Build(indices, mesh.VertexCount);
        int[] degree = new int[faces + 1];
        bool[] open = new bool[faces];
        for (int s = 0; s < table.Count;)
        {
            int e = table.RunEnd(s);
            if (e - s == 2)
            {
                degree[table.HalfEdges[s] / 3]++;
                degree[table.HalfEdges[s + 1] / 3]++;
            }
            else
            {
                for (int k = s; k < e; k++) open[table.HalfEdges[k] / 3] = true;
            }
            s = e;
        }
        int[] start = new int[faces + 1];
        for (int f = 0; f < faces; f++) start[f + 1] = start[f] + degree[f];
        int[] neighbor = new int[start[faces]];
        bool[] sameDirection = new bool[start[faces]];
        int[] fill = new int[faces];
        for (int s = 0; s < table.Count;)
        {
            int e = table.RunEnd(s);
            if (e - s == 2)
            {
                int h0 = table.HalfEdges[s], h1 = table.HalfEdges[s + 1];
                int f0 = h0 / 3, f1 = h1 / 3;
                bool same = table.From(h0) == table.From(h1);
                neighbor[start[f0] + fill[f0]] = f1; sameDirection[start[f0] + fill[f0]++] = same;
                neighbor[start[f1] + fill[f1]] = f0; sameDirection[start[f1] + fill[f1]++] = same;
            }
            s = e;
        }

        bool[] flip = new bool[faces];
        int[] component = new int[faces];
        Array.Fill(component, -1);
        List<List<int>> components = [];
        Queue<int> queue = new();
        for (int seed = 0; seed < faces; seed++)
        {
            if (component[seed] >= 0) continue;
            List<int> members = [];
            component[seed] = components.Count;
            queue.Enqueue(seed);
            while (queue.Count > 0)
            {
                int f = queue.Dequeue();
                members.Add(f);
                for (int k = start[f]; k < start[f + 1]; k++)
                {
                    int g = neighbor[k];
                    if (component[g] >= 0) continue;
                    component[g] = components.Count;
                    flip[g] = sameDirection[k] ? !flip[f] : flip[f];
                    queue.Enqueue(g);
                }
            }
            components.Add(members);
        }

        foreach (List<int> members in components)
        {
            bool closed = true;
            foreach (int f in members)
                if (open[f]) { closed = false; break; }
            if (orientOutward && closed)
            {
                double volume = 0;
                foreach (int f in members)
                    volume += SignedVolume(mesh.Vertices, indices, f, flip[f]);
                if (volume < 0)
                    foreach (int f in members) flip[f] = !flip[f];
            }
        }
        for (int f = 0; f < faces; f++)
            if (flip[f]) (indices[f * 3 + 1], indices[f * 3 + 2]) = (indices[f * 3 + 2], indices[f * 3 + 1]);
        return Compact(mesh.Vertices, indices);
    }

    // The two half-edges of a run to keep joined: a consistently wound pair first, then the largest faces.
    private static (int I, int J) BestPair(Mesh mesh, EdgeTable table, List<int> run)
    {
        int bestI = 0, bestJ = 1;
        bool bestConsistent = false;
        double bestArea = double.NegativeInfinity;
        for (int i = 0; i < run.Count; i++)
        {
            for (int j = i + 1; j < run.Count; j++)
            {
                // Compared in this order: a bonus added to the area (the earlier 1e30) swallows the area in double precision.
                bool consistent = table.From(run[i]) != table.From(run[j]);
                double area = FaceAreaOf(mesh, run[i] / 3) + FaceAreaOf(mesh, run[j] / 3);
                if (consistent != bestConsistent ? consistent : area > bestArea)
                {
                    bestConsistent = consistent;
                    bestArea = area;
                    bestI = i;
                    bestJ = j;
                }
            }
        }
        return (bestI, bestJ);
    }

    private static void ThinRun(Mesh mesh, EdgeTable table, List<int> run, bool[] keep)
    {
        while (run.Count > 2)
        {
            (int bestI, int bestJ) = BestPair(mesh, table, run);
            for (int k = run.Count - 1; k >= 0; k--)
            {
                if (k == bestI || k == bestJ) continue;
                keep[run[k] / 3] = false;
                run.RemoveAt(k);
                if (k < bestI) bestI--;
                if (k < bestJ) bestJ--;
            }
        }
    }

    private static (int, int, int) SortedTriple(int a, int b, int c)
    {
        int lo = Math.Min(a, Math.Min(b, c)), hi = Math.Max(a, Math.Max(b, c));
        return (lo, a + b + c - lo - hi, hi);
    }

    private static HashSet<(int, int, int)> FaceSet(int[] indices)
    {
        HashSet<(int, int, int)> set = new(indices.Length / 3);
        for (int f = 0; f < indices.Length / 3; f++)
            set.Add(SortedTriple(indices[f * 3], indices[f * 3 + 1], indices[f * 3 + 2]));
        return set;
    }

    private static void AddFill(List<float> verts, List<int> idx, List<int> loop)
    {
        int n = loop.Count;
        if (n == 3)
        {
            idx.Add(loop[0]); idx.Add(loop[2]); idx.Add(loop[1]);
            return;
        }
        float cx = 0, cy = 0, cz = 0;
        foreach (int v in loop)
        {
            cx += verts[v * 3]; cy += verts[v * 3 + 1]; cz += verts[v * 3 + 2];
        }
        int center = verts.Count / 3;
        verts.Add(cx / n); verts.Add(cy / n); verts.Add(cz / n);
        for (int i = 0; i < n; i++)
        {
            idx.Add(loop[(i + 1) % n]); idx.Add(loop[i]); idx.Add(center);
        }
    }

    private static double LoopPerimeter(float[] v, List<int> loop)
    {
        double sum = 0;
        for (int i = 0; i < loop.Count; i++)
        {
            int a = loop[i] * 3, b = loop[(i + 1) % loop.Count] * 3;
            double dx = v[a] - v[b], dy = v[a + 1] - v[b + 1], dz = v[a + 2] - v[b + 2];
            sum += Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
        return sum;
    }

    private static double SignedVolume(float[] v, int[] idx, int face, bool flipped)
    {
        int a = idx[face * 3] * 3, b = idx[face * 3 + (flipped ? 2 : 1)] * 3, c = idx[face * 3 + (flipped ? 1 : 2)] * 3;
        double ax = v[a], ay = v[a + 1], az = v[a + 2], bx = v[b], by = v[b + 1], bz = v[b + 2], cx = v[c], cy = v[c + 1], cz = v[c + 2];
        return (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6.0;
    }

    private static double FaceAreaOf(Mesh mesh, int face) =>
        FaceArea(mesh.Vertices, mesh.Indices[face * 3], mesh.Indices[face * 3 + 1], mesh.Indices[face * 3 + 2]);

    private static double FaceArea(float[] v, int a, int b, int c)
    {
        double ux = v[b * 3] - (double)v[a * 3], uy = v[b * 3 + 1] - (double)v[a * 3 + 1], uz = v[b * 3 + 2] - (double)v[a * 3 + 2];
        double wx = v[c * 3] - (double)v[a * 3], wy = v[c * 3 + 1] - (double)v[a * 3 + 1], wz = v[c * 3 + 2] - (double)v[a * 3 + 2];
        double cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
        return 0.5 * Math.Sqrt(cx * cx + cy * cy + cz * cz);
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
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra; else parent[ra] = rb;
    }

    private static Mesh Rebuild(Mesh mesh, bool[] keepFace)
    {
        int count = 0;
        foreach (bool k in keepFace) if (k) count++;
        int[] indices = new int[count * 3];
        int o = 0;
        for (int f = 0; f < keepFace.Length; f++)
        {
            if (!keepFace[f]) continue;
            indices[o++] = mesh.Indices[f * 3]; indices[o++] = mesh.Indices[f * 3 + 1]; indices[o++] = mesh.Indices[f * 3 + 2];
        }
        return Compact(mesh.Vertices, indices);
    }

    private static Mesh Compact(float[] vertices, int[] indices)
    {
        int count = vertices.Length / 3;
        int[] map = new int[count];
        Array.Fill(map, -1);
        int kept = 0;
        foreach (int i in indices)
            if (map[i] < 0) map[i] = 0;
        for (int i = 0; i < count; i++)
            if (map[i] == 0) map[i] = kept++;
        float[] outV = new float[kept * 3];
        for (int i = 0; i < count; i++)
        {
            if (map[i] < 0) continue;
            outV[map[i] * 3] = vertices[i * 3]; outV[map[i] * 3 + 1] = vertices[i * 3 + 1]; outV[map[i] * 3 + 2] = vertices[i * 3 + 2];
        }
        int[] outI = new int[indices.Length];
        for (int i = 0; i < indices.Length; i++)
            outI[i] = map[indices[i]];
        return new Mesh { Vertices = outV, Indices = outI };
    }

    private static void Validate(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (mesh.Vertices.Length % 3 != 0 || mesh.Indices.Length % 3 != 0)
            throw new ArgumentException("Mesh arrays must hold whole vertices and triangles.", nameof(mesh));
        int count = mesh.VertexCount;
        foreach (int i in mesh.Indices)
            if ((uint)i >= (uint)count)
                throw new ArgumentException("A triangle index lies outside the vertex array.", nameof(mesh));
    }
}
