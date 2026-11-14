namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Garland-Heckbert quadric edge-collapse state over an indexed triangle mesh. Collapses keep the mesh manifold (link
/// condition), keep boundaries pinned and refuse fold-overs; ties break on edge keys, so results are deterministic.</summary>
internal sealed class QuadricEdgeCollapser
{
    private readonly MeshDecimatorOptions _options;
    private readonly double[] _pos;
    private readonly double[] _quadric;
    private readonly int[] _faces;
    private readonly bool[] _faceAlive;
    private readonly List<int>[] _vertexFaces;
    private readonly bool[] _vertexAlive;
    private readonly bool[] _boundary;
    private readonly bool[] _locked;
    private readonly int[] _version;
    private readonly HashSet<int> _scratchA = [];
    private readonly HashSet<int> _scratchB = [];
    private readonly HashSet<long> _scratchEdges = [];
    private int _aliveFaces;

    public QuadricEdgeCollapser(Mesh mesh, MeshDecimatorOptions options)
    {
        _options = options;
        int vertexCount = mesh.VertexCount;
        _pos = new double[vertexCount * 3];
        for (int i = 0; i < _pos.Length; i++)
            _pos[i] = mesh.Vertices[i];
        _quadric = new double[vertexCount * 10];
        _faces = (int[])mesh.Indices.Clone();
        int faceCount = mesh.TriangleCount;
        _faceAlive = new bool[faceCount];
        Array.Fill(_faceAlive, true);
        _aliveFaces = faceCount;
        _vertexFaces = new List<int>[vertexCount];
        for (int i = 0; i < vertexCount; i++)
            _vertexFaces[i] = new List<int>(6);
        _vertexAlive = new bool[vertexCount];
        Array.Fill(_vertexAlive, true);
        _boundary = new bool[vertexCount];
        _locked = new bool[vertexCount];
        _version = new int[vertexCount];

        for (int f = 0; f < faceCount; f++)
        {
            for (int c = 0; c < 3; c++)
                _vertexFaces[_faces[f * 3 + c]].Add(f);
            AddFaceQuadric(f);
        }
        MarkBoundaries(vertexCount);
    }

    public int AliveFaces => _aliveFaces;

    /// <summary>Collapses edges cheapest-first until at most <paramref name="target"/> faces remain or no legal collapse is left.</summary>
    public void Run(int target, CancellationToken cancellationToken)
    {
        PriorityQueue<Candidate, (double, long)> heap = new();
        for (int f = 0; f < _faces.Length / 3; f++)
            for (int c = 0; c < 3; c++)
            {
                int a = _faces[f * 3 + c], b = _faces[f * 3 + (c + 1) % 3];
                if (a < b || !EdgeHasOtherFace(a, b, f))
                    Push(heap, Math.Min(a, b), Math.Max(a, b));
            }

        int pops = 0;
        HashSet<int> touched = [];
        while (_aliveFaces > target && heap.TryDequeue(out Candidate cand, out (double Cost, long Key) _))
        {
            if ((++pops & 1023) == 0)
                cancellationToken.ThrowIfCancellationRequested();
            if (!_vertexAlive[cand.A] || !_vertexAlive[cand.B] || _version[cand.A] != cand.VersionA || _version[cand.B] != cand.VersionB)
                continue;
            if (!TryPlace(cand.A, cand.B, out double x, out double y, out double z, out _) || !IsLegal(cand.A, cand.B, x, y, z))
                continue;
            Collapse(cand.A, cand.B, x, y, z);
            touched.Clear();
            foreach (int f in _vertexFaces[cand.A])
                for (int c = 0; c < 3; c++)
                {
                    int v = _faces[f * 3 + c];
                    if (v != cand.A && touched.Add(v))
                        Push(heap, Math.Min(cand.A, v), Math.Max(cand.A, v));
                }
        }
    }

    public Mesh ToMesh()
    {
        int[] map = new int[_vertexAlive.Length];
        Array.Fill(map, -1);
        List<int> indices = new(_aliveFaces * 3);
        for (int f = 0; f < _faceAlive.Length; f++)
            if (_faceAlive[f])
                for (int c = 0; c < 3; c++) map[_faces[f * 3 + c]] = 0;
        int kept = 0;
        for (int i = 0; i < map.Length; i++)
            if (map[i] == 0) map[i] = kept++;
        float[] vertices = new float[kept * 3];
        for (int i = 0; i < map.Length; i++)
        {
            if (map[i] < 0) continue;
            vertices[map[i] * 3] = (float)_pos[i * 3];
            vertices[map[i] * 3 + 1] = (float)_pos[i * 3 + 1];
            vertices[map[i] * 3 + 2] = (float)_pos[i * 3 + 2];
        }
        for (int f = 0; f < _faceAlive.Length; f++)
            if (_faceAlive[f])
                for (int c = 0; c < 3; c++) indices.Add(map[_faces[f * 3 + c]]);
        return new Mesh { Vertices = vertices, Indices = [.. indices] };
    }

    private bool EdgeHasOtherFace(int a, int b, int face)
    {
        foreach (int g in _vertexFaces[a])
        {
            if (g == face) continue;
            for (int c = 0; c < 3; c++)
                if (_faces[g * 3 + c] == b) return true;
        }
        return false;
    }

    private void Push(PriorityQueue<Candidate, (double, long)> heap, int a, int b)
    {
        if (_locked[a] || _locked[b] || !TryPlace(a, b, out _, out _, out _, out double cost))
            return;
        heap.Enqueue(new Candidate(a, b, _version[a], _version[b]), (cost, (long)a << 32 | (uint)b));
    }

    private void AddFaceQuadric(int f)
    {
        int a = _faces[f * 3], b = _faces[f * 3 + 1], c = _faces[f * 3 + 2];
        FaceNormal(a, b, c, out double nx, out double ny, out double nz);
        double len = Math.Sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-300) return;
        double area = 0.5 * len;
        nx /= len; ny /= len; nz /= len;
        double d = -(nx * _pos[a * 3] + ny * _pos[a * 3 + 1] + nz * _pos[a * 3 + 2]);
        for (int k = 0; k < 3; k++)
            AddPlane(_faces[f * 3 + k], nx, ny, nz, d, area);
    }

    private void AddPlane(int v, double nx, double ny, double nz, double d, double weight)
    {
        int o = v * 10;
        _quadric[o] += weight * nx * nx; _quadric[o + 1] += weight * nx * ny; _quadric[o + 2] += weight * nx * nz;
        _quadric[o + 3] += weight * nx * d; _quadric[o + 4] += weight * ny * ny; _quadric[o + 5] += weight * ny * nz;
        _quadric[o + 6] += weight * ny * d; _quadric[o + 7] += weight * nz * nz; _quadric[o + 8] += weight * nz * d;
        _quadric[o + 9] += weight * d * d;
    }

    private void MarkBoundaries(int vertexCount)
    {
        EdgeTable table = EdgeTable.Build(_faces, vertexCount);
        for (int s = 0; s < table.Count;)
        {
            int e = table.RunEnd(s);
            int run = e - s;
            if (run != 2)
            {
                int slot = table.HalfEdges[s];
                int a = table.From(slot), b = table.To(slot);
                _boundary[a] = true;
                _boundary[b] = true;
                if (run > 2)
                {
                    _locked[a] = true;
                    _locked[b] = true;
                }
                else
                {
                    AddBoundaryPlane(slot / 3, a, b);
                }
            }
            s = e;
        }
    }

    private void AddBoundaryPlane(int face, int a, int b)
    {
        FaceNormal(_faces[face * 3], _faces[face * 3 + 1], _faces[face * 3 + 2], out double fx, out double fy, out double fz);
        double flen = Math.Sqrt(fx * fx + fy * fy + fz * fz);
        if (flen < 1e-300) return;
        double ex = _pos[b * 3] - _pos[a * 3], ey = _pos[b * 3 + 1] - _pos[a * 3 + 1], ez = _pos[b * 3 + 2] - _pos[a * 3 + 2];
        double mx = ey * fz - ez * fy, my = ez * fx - ex * fz, mz = ex * fy - ey * fx;
        double mlen = Math.Sqrt(mx * mx + my * my + mz * mz);
        if (mlen < 1e-300) return;
        mx /= mlen; my /= mlen; mz /= mlen;
        double d = -(mx * _pos[a * 3] + my * _pos[a * 3 + 1] + mz * _pos[a * 3 + 2]);
        double weight = _options.BoundaryWeight * (ex * ex + ey * ey + ez * ez);
        AddPlane(a, mx, my, mz, d, weight);
        AddPlane(b, mx, my, mz, d, weight);
    }

    private void FaceNormal(int a, int b, int c, out double nx, out double ny, out double nz)
    {
        double ux = _pos[b * 3] - _pos[a * 3], uy = _pos[b * 3 + 1] - _pos[a * 3 + 1], uz = _pos[b * 3 + 2] - _pos[a * 3 + 2];
        double wx = _pos[c * 3] - _pos[a * 3], wy = _pos[c * 3 + 1] - _pos[a * 3 + 1], wz = _pos[c * 3 + 2] - _pos[a * 3 + 2];
        nx = uy * wz - uz * wy; ny = uz * wx - ux * wz; nz = ux * wy - uy * wx;
    }

    private int SharedFaces(int a, int b, Span<int> shared)
    {
        int count = 0;
        foreach (int f in _vertexFaces[a])
        {
            if (_faces[f * 3] != b && _faces[f * 3 + 1] != b && _faces[f * 3 + 2] != b) continue;
            if (count < shared.Length) shared[count] = f;
            count++;
        }
        return count;
    }

    private bool TryPlace(int a, int b, out double x, out double y, out double z, out double cost)
    {
        x = y = z = 0;
        cost = double.MaxValue;
        bool ba = _boundary[a], bb = _boundary[b];
        Span<double> q = stackalloc double[10];
        for (int i = 0; i < 10; i++)
            q[i] = _quadric[a * 10 + i] + _quadric[b * 10 + i];
        double ax = _pos[a * 3], ay = _pos[a * 3 + 1], az = _pos[a * 3 + 2];
        double bx = _pos[b * 3], by = _pos[b * 3 + 1], bz = _pos[b * 3 + 2];
        double mx = 0.5 * (ax + bx), my = 0.5 * (ay + by), mz = 0.5 * (az + bz);
        double len2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz);

        // A boundary vertex never moves unless the edge joins two boundary vertices (a pinch, only with AllowBoundaryPinch): collapsing into it keeps the boundary curve; the ends of a boundary edge may also meet midway.
        if (!ba || bb) Consider(q, bx, by, bz, ref x, ref y, ref z, ref cost);
        if (!bb || ba) Consider(q, ax, ay, az, ref x, ref y, ref z, ref cost);
        if (!ba && !bb)
        {
            Consider(q, mx, my, mz, ref x, ref y, ref z, ref cost);
            if (SolveOptimal(q, out double ox, out double oy, out double oz)
                && (ox - mx) * (ox - mx) + (oy - my) * (oy - my) + (oz - mz) * (oz - mz) <= 2.25 * len2)
                Consider(q, ox, oy, oz, ref x, ref y, ref z, ref cost);
        }
        else if (ba && bb)
        {
            Span<int> shared = stackalloc int[3];
            if (SharedFaces(a, b, shared) == 1)
                Consider(q, mx, my, mz, ref x, ref y, ref z, ref cost);
        }
        if (cost == double.MaxValue)
            return false;
        cost += _options.EdgeLengthWeight * len2;
        return true;
    }

    private static void Consider(ReadOnlySpan<double> q, double px, double py, double pz, ref double bx, ref double by, ref double bz,
        ref double bestCost)
    {
        double c = Error(q, px, py, pz);
        if (c < bestCost)
        {
            bestCost = c; bx = px; by = py; bz = pz;
        }
    }

    private static double Error(ReadOnlySpan<double> q, double x, double y, double z) =>
        q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y
        + q[7] * z * z + 2 * q[8] * z + q[9];

    private static bool SolveOptimal(ReadOnlySpan<double> q, out double x, out double y, out double z)
    {
        x = y = z = 0;
        double a = q[0], b = q[1], c = q[2], e = q[4], f = q[5], i = q[7];
        double det = a * (e * i - f * f) - b * (b * i - f * c) + c * (b * f - e * c);
        double scale = Math.Abs(a) + Math.Abs(e) + Math.Abs(i);
        if (scale <= 0 || Math.Abs(det) < 1e-9 * scale * scale * scale)
            return false;
        double r0 = -q[3], r1 = -q[6], r2 = -q[8];
        x = (r0 * (e * i - f * f) - b * (r1 * i - f * r2) + c * (r1 * f - e * r2)) / det;
        y = (a * (r1 * i - f * r2) - r0 * (b * i - f * c) + c * (b * r2 - r1 * c)) / det;
        z = (a * (e * r2 - r1 * f) - b * (b * r2 - r1 * c) + r0 * (b * f - e * c)) / det;
        return double.IsFinite(x) && double.IsFinite(y) && double.IsFinite(z);
    }

    private bool IsLegal(int a, int b, double x, double y, double z)
    {
        Span<int> shared = stackalloc int[3];
        int sharedCount = SharedFaces(a, b, shared);
        if (sharedCount is < 1 or > 2)
            return false;
        if (_boundary[a] && _boundary[b] && sharedCount == 2 && !_options.AllowBoundaryPinch)
            return false;

        _scratchA.Clear();
        _scratchB.Clear();
        foreach (int f in _vertexFaces[a])
            for (int c = 0; c < 3; c++)
                if (_faces[f * 3 + c] != a) _scratchA.Add(_faces[f * 3 + c]);
        foreach (int f in _vertexFaces[b])
            for (int c = 0; c < 3; c++)
                if (_faces[f * 3 + c] != b) _scratchB.Add(_faces[f * 3 + c]);
        int common = 0;
        foreach (int v in _scratchA)
            if (_scratchB.Contains(v)) common++;
        if (common != sharedCount)
            return false;

        // The full link condition also forbids a link edge on both sides: a face (a, c, d) and a face (b, c, d) would merge into two
        // coincident triangles. Counting common neighbours alone misses this, e.g. on a tetrahedron.
        if (WouldDuplicateFace(a, b))
            return false;

        return NormalsHold(a, x, y, z, shared, sharedCount) && NormalsHold(b, x, y, z, shared, sharedCount);
    }

    /// <summary>True when a face around <paramref name="a"/> and a face around <paramref name="b"/>, neither holding the other vertex,
    /// share the same two remaining vertices, so collapsing the edge would leave two faces on one vertex pair.</summary>
    private bool WouldDuplicateFace(int a, int b)
    {
        _scratchEdges.Clear();
        foreach (int f in _vertexFaces[a])
        {
            if (FaceHas(f, b)) continue;
            _scratchEdges.Add(OppositeEdgeKey(f, a));
        }
        foreach (int f in _vertexFaces[b])
        {
            if (FaceHas(f, a)) continue;
            if (_scratchEdges.Contains(OppositeEdgeKey(f, b))) return true;
        }
        return false;
    }

    private bool FaceHas(int face, int vertex) =>
        _faces[face * 3] == vertex || _faces[face * 3 + 1] == vertex || _faces[face * 3 + 2] == vertex;

    /// <summary>Order-independent key of the two vertices of <paramref name="face"/> other than <paramref name="vertex"/>.</summary>
    private long OppositeEdgeKey(int face, int vertex)
    {
        int first = -1, second = -1;
        for (int c = 0; c < 3; c++)
        {
            int v = _faces[face * 3 + c];
            if (v == vertex) continue;
            if (first < 0) first = v; else second = v;
        }
        return ((long)Math.Min(first, second) << 32) | (uint)Math.Max(first, second);
    }

    private bool NormalsHold(int moving, double x, double y, double z, ReadOnlySpan<int> shared, int sharedCount)
    {
        foreach (int f in _vertexFaces[moving])
        {
            bool isShared = false;
            for (int k = 0; k < sharedCount; k++)
                if (shared[k] == f) isShared = true;
            if (isShared) continue;
            int a = _faces[f * 3], b = _faces[f * 3 + 1], c = _faces[f * 3 + 2];
            FaceNormal(a, b, c, out double ox, out double oy, out double oz);
            double px0 = _pos[moving * 3], py0 = _pos[moving * 3 + 1], pz0 = _pos[moving * 3 + 2];
            _pos[moving * 3] = x; _pos[moving * 3 + 1] = y; _pos[moving * 3 + 2] = z;
            FaceNormal(a, b, c, out double nx, out double ny, out double nz);
            _pos[moving * 3] = px0; _pos[moving * 3 + 1] = py0; _pos[moving * 3 + 2] = pz0;
            double oldLen = Math.Sqrt(ox * ox + oy * oy + oz * oz), newLen = Math.Sqrt(nx * nx + ny * ny + nz * nz);
            if (oldLen < 1e-300)
                continue;
            if (newLen < 1e-12 * oldLen)
                return false;
            if ((ox * nx + oy * ny + oz * nz) / (oldLen * newLen) < _options.MinNormalDot)
                return false;
        }
        return true;
    }

    private void Collapse(int a, int b, double x, double y, double z)
    {
        Span<int> shared = stackalloc int[3];
        int sharedCount = SharedFaces(a, b, shared);
        for (int k = 0; k < sharedCount; k++)
        {
            int f = shared[k];
            _faceAlive[f] = false;
            _aliveFaces--;
            for (int c = 0; c < 3; c++)
                _vertexFaces[_faces[f * 3 + c]].Remove(f);
        }
        foreach (int f in _vertexFaces[b])
        {
            for (int c = 0; c < 3; c++)
                if (_faces[f * 3 + c] == b) _faces[f * 3 + c] = a;
            _vertexFaces[a].Add(f);
        }
        _vertexFaces[b].Clear();
        _vertexAlive[b] = false;
        _pos[a * 3] = x; _pos[a * 3 + 1] = y; _pos[a * 3 + 2] = z;
        for (int i = 0; i < 10; i++)
            _quadric[a * 10 + i] += _quadric[b * 10 + i];
        _boundary[a] |= _boundary[b];
        _locked[a] |= _locked[b];
        _version[a]++;
        _version[b]++;
    }

    private readonly record struct Candidate(int A, int B, int VersionA, int VersionB);
}
