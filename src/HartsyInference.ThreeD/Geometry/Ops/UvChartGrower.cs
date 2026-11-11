namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Grows one planar chart at a time over a triangle mesh and keeps it valid as it grows. A chart has a projection plane (its normal and a
/// right-handed tangent basis); a face joins it only when its normal stays within the cone of the plane normal and its projected triangle
/// overlaps none of the chart's triangles, so no chart ever needs to be split afterwards. A projection is an orthogonal projection of the
/// 3D positions, so a face whose normal is within the cone (under 90 degrees) cannot be mirrored and overlap is the only fold-over left to test.
/// Faces are taken best first (smallest normal deviation, lowest index on ties), so the result does not depend on hash or sort order.
/// </summary>
internal sealed class UvChartGrower
{
    private const int MaxCellSpan = 32;

    private readonly Mesh _mesh;
    private readonly double[] _normal;
    private readonly double[] _area;
    private readonly int[] _neighbor;
    private readonly double _cell;
    private readonly int[] _rejectedIn;
    private readonly int[] _seenIn;
    private readonly Dictionary<long, int> _head = [];
    private readonly List<int> _next = [];
    private readonly List<int> _item = [];
    private readonly List<int> _big = [];
    private readonly double[] _corners = new double[6], _otherCorners = new double[6];
    private int _rejectToken, _seenToken;
    private double _tx, _ty, _tz, _bx, _by, _bz;

    /// <param name="cell">Edge of the hash cells in projected units; about twice the mean edge length keeps a handful of triangles per cell.</param>
    public UvChartGrower(Mesh mesh, double[] normal, double[] area, int[] neighbor, double cell)
    {
        _mesh = mesh;
        _normal = normal;
        _area = area;
        _neighbor = neighbor;
        _cell = Math.Max(cell, 1e-300);
        _rejectedIn = new int[mesh.TriangleCount];
        _seenIn = new int[mesh.TriangleCount];
    }

    /// <summary>The unit normal a chart seeded at <paramref name="seed"/> projects along: the area-weighted mean of the seed face and its
    /// edge neighbours, so one noisy facet does not tilt a whole chart. Falls back to the seed's own normal.</summary>
    public double[] SeedNormal(int seed)
    {
        double x = _normal[seed * 3] * _area[seed], y = _normal[seed * 3 + 1] * _area[seed], z = _normal[seed * 3 + 2] * _area[seed];
        for (int c = 0; c < 3; c++)
        {
            int g = _neighbor[seed * 3 + c];
            if (g < 0)
                continue;
            x += _normal[g * 3] * _area[g]; y += _normal[g * 3 + 1] * _area[g]; z += _normal[g * 3 + 2] * _area[g];
        }

        double len = Math.Sqrt(x * x + y * y + z * z);
        if (len < 1e-300)
            return [_normal[seed * 3], _normal[seed * 3 + 1], _normal[seed * 3 + 2]];
        return [x / len, y / len, z / len];
    }

    /// <summary>The right-handed tangent basis <c>(t, b)</c> of a plane with unit normal <paramref name="n"/>; the helper axis least aligned
    /// with the normal keeps it well conditioned.</summary>
    public static void Basis(double[] n, out double[] t, out double[] b)
    {
        double ax = n[0], ay = n[1], az = n[2];
        double hx = 0, hy = 0, hz = 0;
        if (Math.Abs(ax) <= Math.Abs(ay) && Math.Abs(ax) <= Math.Abs(az)) hx = 1;
        else if (Math.Abs(ay) <= Math.Abs(az)) hy = 1;
        else hz = 1;
        double tx = hy * az - hz * ay, ty = hz * ax - hx * az, tz = hx * ay - hy * ax;
        double tl = Math.Sqrt(tx * tx + ty * ty + tz * tz);
        tx /= tl; ty /= tl; tz /= tl;
        t = [tx, ty, tz];
        b = [ay * tz - az * ty, az * tx - ax * tz, ax * ty - ay * tx];
    }

    /// <summary>Grows a chart from <paramref name="faces"/> (one seed, or the faces of a chart being extended) along <paramref name="normal"/>,
    /// taking faces that <paramref name="free"/> accepts and whose normal has a dot product with <paramref name="normal"/> of at least
    /// <paramref name="coneCos"/>. Appends the faces it takes to <paramref name="faces"/> and calls <paramref name="taken"/> for each.</summary>
    public void Grow(List<int> faces, double[] normal, double coneCos, Func<int, bool> free, Action<int> taken, CancellationToken cancellationToken)
    {
        Basis(normal, out double[] t, out double[] b);
        _tx = t[0]; _ty = t[1]; _tz = t[2]; _bx = b[0]; _by = b[1]; _bz = b[2];
        _head.Clear();
        _next.Clear();
        _item.Clear();
        _big.Clear();
        int rejectToken = ++_rejectToken;
        foreach (int f in faces)
            Insert(f);

        PriorityQueue<int, (double Deviation, int Face)> frontier = new();
        foreach (int f in faces)
            Push(f);

        int processed = 0;
        while (frontier.TryDequeue(out int g, out _))
        {
            if ((++processed & 1023) == 0)
                cancellationToken.ThrowIfCancellationRequested();
            if (_rejectedIn[g] == rejectToken || !free(g))
                continue;
            double dot = _normal[g * 3] * normal[0] + _normal[g * 3 + 1] * normal[1] + _normal[g * 3 + 2] * normal[2];
            if (dot < coneCos || (_area[g] > 0 && Overlaps(g)))
            {
                _rejectedIn[g] = rejectToken;
                continue;
            }

            faces.Add(g);
            taken(g);
            Insert(g);
            Push(g);
        }

        void Push(int f)
        {
            for (int c = 0; c < 3; c++)
            {
                int g = _neighbor[f * 3 + c];
                if (g < 0 || _rejectedIn[g] == rejectToken || !free(g))
                    continue;
                double dot = _normal[g * 3] * normal[0] + _normal[g * 3 + 1] * normal[1] + _normal[g * 3 + 2] * normal[2];
                frontier.Enqueue(g, (1.0 - dot, g));
            }
        }
    }

    // Projected corners of face f: (u0, v0, u1, v1, u2, v2).
    private void Corners(int f, double[] uv)
    {
        for (int k = 0; k < 3; k++)
        {
            int v = _mesh.Indices[f * 3 + k] * 3;
            double x = _mesh.Vertices[v], y = _mesh.Vertices[v + 1], z = _mesh.Vertices[v + 2];
            uv[k * 2] = x * _tx + y * _ty + z * _tz;
            uv[k * 2 + 1] = x * _bx + y * _by + z * _bz;
        }
    }

    private bool Overlaps(int g)
    {
        double[] a = _corners, other = _otherCorners;
        Corners(g, a);
        double ax0 = Math.Min(a[0], Math.Min(a[2], a[4])), ax1 = Math.Max(a[0], Math.Max(a[2], a[4]));
        double ay0 = Math.Min(a[1], Math.Min(a[3], a[5])), ay1 = Math.Max(a[1], Math.Max(a[3], a[5]));
        double sliver = 1e-14 * _cell * _cell;
        if (0.5 * Math.Abs((a[2] - a[0]) * (a[5] - a[1]) - (a[4] - a[0]) * (a[3] - a[1])) <= sliver)
            return false;
        double eps = 1e-7 * _cell;
        int token = ++_seenToken;
        foreach (int t in _big)
        {
            if (Test(t))
                return true;
        }

        Range(ax0, ay0, ax1, ay1, out int cx0, out int cy0, out int cx1, out int cy1);
        if (cx1 - cx0 >= MaxCellSpan || cy1 - cy0 >= MaxCellSpan)
            return ScanAll(g, eps);
        for (int cy = cy0; cy <= cy1; cy++)
        {
            for (int cx = cx0; cx <= cx1; cx++)
            {
                if (!_head.TryGetValue(Key(cx, cy), out int e))
                    continue;
                for (; e >= 0; e = _next[e])
                {
                    if (Test(_item[e]))
                        return true;
                }
            }
        }

        return false;

        bool Test(int t)
        {
            if (t == g || _seenIn[t] == token)
                return false;
            _seenIn[t] = token;
            Corners(t, other);
            if (0.5 * Math.Abs((other[2] - other[0]) * (other[5] - other[1]) - (other[4] - other[0]) * (other[3] - other[1])) <= sliver)
                return false;
            return UvChartValidator.TrianglesOverlap(a, other, eps);
        }
    }

    // A face wider than the grid is meant for: compare it with every triangle of the chart.
    private bool ScanAll(int g, double eps)
    {
        double[] a = _corners, other = _otherCorners;
        double sliver = 1e-14 * _cell * _cell;
        foreach (KeyValuePair<long, int> cell in _head)
        {
            for (int e = cell.Value; e >= 0; e = _next[e])
            {
                int t = _item[e];
                if (t == g)
                    continue;
                Corners(t, other);
                if (0.5 * Math.Abs((other[2] - other[0]) * (other[5] - other[1]) - (other[4] - other[0]) * (other[3] - other[1])) <= sliver)
                    continue;
                if (UvChartValidator.TrianglesOverlap(a, other, eps))
                    return true;
            }
        }

        foreach (int t in _big)
        {
            if (t == g)
                continue;
            Corners(t, other);
            if (UvChartValidator.TrianglesOverlap(a, other, eps))
                return true;
        }

        return false;
    }

    private void Insert(int f)
    {
        double[] a = _corners;
        Corners(f, a);
        Range(Math.Min(a[0], Math.Min(a[2], a[4])), Math.Min(a[1], Math.Min(a[3], a[5])),
            Math.Max(a[0], Math.Max(a[2], a[4])), Math.Max(a[1], Math.Max(a[3], a[5])), out int cx0, out int cy0, out int cx1, out int cy1);
        if (cx1 - cx0 >= MaxCellSpan || cy1 - cy0 >= MaxCellSpan)
        {
            _big.Add(f);
            return;
        }

        for (int cy = cy0; cy <= cy1; cy++)
        {
            for (int cx = cx0; cx <= cx1; cx++)
            {
                long key = Key(cx, cy);
                int previous = _head.TryGetValue(key, out int h) ? h : -1;
                _head[key] = _item.Count;
                _item.Add(f);
                _next.Add(previous);
            }
        }
    }

    private void Range(double x0, double y0, double x1, double y1, out int cx0, out int cy0, out int cx1, out int cy1)
    {
        cx0 = (int)Math.Floor(x0 / _cell); cx1 = (int)Math.Floor(x1 / _cell);
        cy0 = (int)Math.Floor(y0 / _cell); cy1 = (int)Math.Floor(y1 / _cell);
    }

    private static long Key(int cx, int cy) => ((long)cx << 32) ^ (uint)cy;
}
