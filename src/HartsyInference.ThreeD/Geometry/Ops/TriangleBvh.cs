namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Bounding-volume hierarchy over a triangle mesh answering closest-point-on-surface and unsigned-distance queries. The build is
/// a deterministic median split; queries are read-only and safe to run from many threads.
/// </summary>
public sealed class TriangleBvh
{
    private const int LeafSize = 4;
    private const int StackSize = 128;
    private readonly float[] _vertices;
    private readonly int[] _indices;
    private readonly int[] _order;
    private readonly Node[] _nodes;
    private int _nodeCount;

    /// <summary>Builds the hierarchy; the mesh arrays are referenced, not copied, and must not change afterwards.</summary>
    public TriangleBvh(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (mesh.TriangleCount == 0)
            throw new ArgumentException("The mesh has no triangles.", nameof(mesh));
        _vertices = mesh.Vertices;
        _indices = mesh.Indices;
        int count = mesh.TriangleCount;
        _order = new int[count];
        float[] centroids = new float[count * 3];
        for (int t = 0; t < count; t++)
        {
            _order[t] = t;
            for (int a = 0; a < 3; a++)
                centroids[t * 3 + a] = (_vertices[_indices[t * 3] * 3 + a] + _vertices[_indices[t * 3 + 1] * 3 + a]
                    + _vertices[_indices[t * 3 + 2] * 3 + a]) / 3f;
        }
        _nodes = new Node[Math.Max(1, 2 * count)];
        Build(0, count, centroids, new float[count]);
    }

    /// <summary>Number of triangles indexed.</summary>
    public int TriangleCount => _indices.Length / 3;

    /// <summary>Finds the point on the mesh surface nearest to (<paramref name="x"/>, <paramref name="y"/>, <paramref name="z"/>).</summary>
    public BvhHit ClosestPoint(double x, double y, double z)
    {
        Span<int> stack = stackalloc int[StackSize];
        int top = 0;
        stack[top++] = 0;
        double best = double.MaxValue;
        int bestTri = -1;
        double bx = 0, by = 0, bz = 0, bu = 0, bv = 0;
        while (top > 0)
        {
            int n = stack[--top];
            ref readonly Node node = ref _nodes[n];
            if (BoxDistanceSquared(in node, x, y, z) >= best)
                continue;
            if (node.Count > 0)
            {
                for (int i = node.Left; i < node.Left + node.Count; i++)
                {
                    int tri = _order[i];
                    double d = TriangleClosest(tri, x, y, z, out double px, out double py, out double pz, out double u, out double v);
                    if (d < best || (d == best && tri < bestTri))
                    {
                        best = d; bestTri = tri; bx = px; by = py; bz = pz; bu = u; bv = v;
                    }
                }
                continue;
            }
            double dl = BoxDistanceSquared(in _nodes[node.Left], x, y, z);
            double dr = BoxDistanceSquared(in _nodes[node.Right], x, y, z);
            if (dl <= dr)
            {
                stack[top++] = node.Right;
                stack[top++] = node.Left;
            }
            else
            {
                stack[top++] = node.Left;
                stack[top++] = node.Right;
            }
        }
        return new BvhHit(bestTri, bx, by, bz, best, bu, bv);
    }

    /// <summary>Unsigned distance from the point to the surface.</summary>
    public double Distance(double x, double y, double z) => ClosestPoint(x, y, z).Distance;

    private int Build(int start, int count, float[] centroids, float[] keys)
    {
        int nodeIndex = _nodeCount++;
        float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
        float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
        float cMinX = float.MaxValue, cMinY = float.MaxValue, cMinZ = float.MaxValue;
        float cMaxX = float.MinValue, cMaxY = float.MinValue, cMaxZ = float.MinValue;
        for (int i = start; i < start + count; i++)
        {
            int tri = _order[i];
            for (int k = 0; k < 3; k++)
            {
                int vi = _indices[tri * 3 + k] * 3;
                minX = MathF.Min(minX, _vertices[vi]); maxX = MathF.Max(maxX, _vertices[vi]);
                minY = MathF.Min(minY, _vertices[vi + 1]); maxY = MathF.Max(maxY, _vertices[vi + 1]);
                minZ = MathF.Min(minZ, _vertices[vi + 2]); maxZ = MathF.Max(maxZ, _vertices[vi + 2]);
            }
            float cx = centroids[tri * 3], cy = centroids[tri * 3 + 1], cz = centroids[tri * 3 + 2];
            cMinX = MathF.Min(cMinX, cx); cMaxX = MathF.Max(cMaxX, cx);
            cMinY = MathF.Min(cMinY, cy); cMaxY = MathF.Max(cMaxY, cy);
            cMinZ = MathF.Min(cMinZ, cz); cMaxZ = MathF.Max(cMaxZ, cz);
        }
        _nodes[nodeIndex] = new Node { MinX = minX, MinY = minY, MinZ = minZ, MaxX = maxX, MaxY = maxY, MaxZ = maxZ };
        if (count <= LeafSize)
        {
            _nodes[nodeIndex].Left = start;
            _nodes[nodeIndex].Count = count;
            return nodeIndex;
        }

        float ex = cMaxX - cMinX, ey = cMaxY - cMinY, ez = cMaxZ - cMinZ;
        int axis = ex >= ey && ex >= ez ? 0 : (ey >= ez ? 1 : 2);
        for (int i = start; i < start + count; i++)
            keys[i] = centroids[_order[i] * 3 + axis];
        Array.Sort(keys, _order, start, count);
        int half = count / 2;
        int left = Build(start, half, centroids, keys);
        int right = Build(start + half, count - half, centroids, keys);
        _nodes[nodeIndex].Left = left;
        _nodes[nodeIndex].Right = right;
        return nodeIndex;
    }

    private static double BoxDistanceSquared(in Node n, double x, double y, double z)
    {
        double dx = x < n.MinX ? n.MinX - x : (x > n.MaxX ? x - n.MaxX : 0.0);
        double dy = y < n.MinY ? n.MinY - y : (y > n.MaxY ? y - n.MaxY : 0.0);
        double dz = z < n.MinZ ? n.MinZ - z : (z > n.MaxZ ? z - n.MaxZ : 0.0);
        return dx * dx + dy * dy + dz * dz;
    }

    private double TriangleClosest(int tri, double px, double py, double pz, out double cx, out double cy, out double cz,
        out double outU, out double outV)
    {
        int ia = _indices[tri * 3] * 3, ib = _indices[tri * 3 + 1] * 3, ic = _indices[tri * 3 + 2] * 3;
        double ax = _vertices[ia], ay = _vertices[ia + 1], az = _vertices[ia + 2];
        double abx = _vertices[ib] - ax, aby = _vertices[ib + 1] - ay, abz = _vertices[ib + 2] - az;
        double acx = _vertices[ic] - ax, acy = _vertices[ic + 1] - ay, acz = _vertices[ic + 2] - az;
        double apx = px - ax, apy = py - ay, apz = pz - az;
        double d1 = abx * apx + aby * apy + abz * apz;
        double d2 = acx * apx + acy * apy + acz * apz;
        double u, v;
        if (d1 <= 0 && d2 <= 0)
        {
            u = 0; v = 0;
        }
        else
        {
            double bpx = px - (ax + abx), bpy = py - (ay + aby), bpz = pz - (az + abz);
            double d3 = abx * bpx + aby * bpy + abz * bpz;
            double d4 = acx * bpx + acy * bpy + acz * bpz;
            double cpx = px - (ax + acx), cpy = py - (ay + acy), cpz = pz - (az + acz);
            double d5 = abx * cpx + aby * cpy + abz * cpz;
            double d6 = acx * cpx + acy * cpy + acz * cpz;
            double vc = d1 * d4 - d3 * d2;
            double vb = d5 * d2 - d1 * d6;
            double va = d3 * d6 - d5 * d4;
            if (d3 >= 0 && d4 <= d3)
            {
                u = 1; v = 0;
            }
            else if (d6 >= 0 && d5 <= d6)
            {
                u = 0; v = 1;
            }
            else if (vc <= 0 && d1 >= 0 && d3 <= 0 && d1 - d3 > 0)
            {
                u = d1 / (d1 - d3); v = 0;
            }
            else if (vb <= 0 && d2 >= 0 && d6 <= 0 && d2 - d6 > 0)
            {
                u = 0; v = d2 / (d2 - d6);
            }
            else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0 && (d4 - d3) + (d5 - d6) > 0)
            {
                double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                u = 1 - w; v = w;
            }
            else
            {
                double denom = va + vb + vc;
                if (Math.Abs(denom) < 1e-300)
                {
                    u = 0; v = 0;
                }
                else
                {
                    u = vb / denom; v = vc / denom;
                }
            }
        }
        cx = ax + abx * u + acx * v;
        cy = ay + aby * u + acy * v;
        cz = az + abz * u + acz * v;
        outU = u;
        outV = v;
        double dx = px - cx, dy = py - cy, dz = pz - cz;
        return dx * dx + dy * dy + dz * dz;
    }

    private struct Node
    {
        public float MinX, MinY, MinZ, MaxX, MaxY, MaxZ;

        /// <summary>Leaf: first slot in the triangle order. Internal: left child.</summary>
        public int Left;

        /// <summary>Internal nodes only: right child.</summary>
        public int Right;

        /// <summary>Triangles in a leaf; zero for internal nodes.</summary>
        public int Count;
    }
}
