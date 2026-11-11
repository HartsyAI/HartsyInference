namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Chart-based UV unwrapping. Faces are grown into charts whose normals stay within a cone of the seed face's normal, each chart is
/// projected onto the plane of its area-weighted mean normal, charts that fold over or overlap are re-split at half the cone angle
/// (down to single triangles), and the charts are rotated, scaled to a uniform texel density and packed into the unit square.
/// Deterministic; the output has the same faces in the same order as the input.
/// </summary>
public static class UvUnwrapper
{
    /// <summary>Unwraps <paramref name="mesh"/> and returns the split-vertex mesh with UVs and the vertex map back to the source. Source normals
    /// (or area-weighted ones computed here) are carried across seams.</summary>
    public static UvUnwrapResult Unwrap(Mesh mesh, UvUnwrapOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        options ??= new UvUnwrapOptions();
        if (mesh.TriangleCount == 0)
            throw new ArgumentException("The mesh has no triangles.", nameof(mesh));
        if (mesh.Vertices.Length % 3 != 0 || mesh.Indices.Length % 3 != 0)
            throw new ArgumentException("Mesh arrays must hold whole vertices and triangles.", nameof(mesh));
        foreach (int i in mesh.Indices)
            if ((uint)i >= (uint)mesh.VertexCount)
                throw new ArgumentException("A triangle index lies outside the vertex array.", nameof(mesh));
        if (options.ConeHalfAngleDegrees is <= 0 or >= 90 || options.MinConeHalfAngleDegrees <= 0 || options.TextureSize <= 0
            || options.PaddingTexels < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Cone angle must be in (0,90) and sizes must be positive.");

        FaceData data = new(mesh);
        List<UvChart> charts = BuildCharts(mesh, data, options, cancellationToken);
        UvAtlasPacker.Pack(charts, (double)options.PaddingTexels / options.TextureSize);
        return Assemble(mesh, charts);
    }

    private static List<UvChart> BuildCharts(Mesh mesh, FaceData data, UvUnwrapOptions options, CancellationToken cancellationToken)
    {
        List<UvChart> charts = [];
        int[] localOf = new int[mesh.VertexCount];
        Array.Fill(localOf, -1);
        List<int> all = new(mesh.TriangleCount);
        for (int f = 0; f < mesh.TriangleCount; f++) all.Add(f);

        Queue<(List<int> Faces, double Angle)> pending = new();
        foreach (List<int> region in data.Segment(all, Math.Cos(options.ConeHalfAngleDegrees * Math.PI / 180.0)))
            pending.Enqueue((region, options.ConeHalfAngleDegrees));

        int processed = 0;
        while (pending.Count > 0)
        {
            if ((++processed & 255) == 0)
                cancellationToken.ThrowIfCancellationRequested();
            (List<int> faces, double angle) = pending.Dequeue();
            UvChart chart = Project(mesh, data, faces, localOf);
            if (faces.Count == 1 || UvChartValidator.IsValid(chart, data.Area))
            {
                charts.Add(chart);
                continue;
            }

            List<List<int>>? parts = null;
            double next = angle;
            while (next * 0.5 >= options.MinConeHalfAngleDegrees)
            {
                next *= 0.5;
                List<List<int>> split = data.Segment(faces, Math.Cos(next * Math.PI / 180.0));
                if (split.Count > 1)
                {
                    parts = split;
                    break;
                }
            }
            if (parts is null)
            {
                foreach (int f in faces)
                    pending.Enqueue(([f], next));
                continue;
            }
            foreach (List<int> part in parts)
                pending.Enqueue((part, next));
        }
        return charts;
    }

    private static UvChart Project(Mesh mesh, FaceData data, List<int> faces, int[] localOf)
    {
        double ax = 0, ay = 0, az = 0, area3d = 0;
        foreach (int f in faces)
        {
            ax += data.Normal[f * 3] * data.Area[f];
            ay += data.Normal[f * 3 + 1] * data.Area[f];
            az += data.Normal[f * 3 + 2] * data.Area[f];
            area3d += data.Area[f];
        }
        double len = Math.Sqrt(ax * ax + ay * ay + az * az);
        if (len < 1e-300)
        {
            ax = data.Normal[faces[0] * 3]; ay = data.Normal[faces[0] * 3 + 1]; az = data.Normal[faces[0] * 3 + 2];
            len = Math.Sqrt(ax * ax + ay * ay + az * az);
        }
        if (len < 1e-300)
        {
            ax = 0; ay = 0; az = 1; len = 1;
        }
        ax /= len; ay /= len; az /= len;

        // Helper vector least aligned with the axis keeps the basis well conditioned; (t, b, axis) is right-handed.
        double hx = 0, hy = 0, hz = 0;
        if (Math.Abs(ax) <= Math.Abs(ay) && Math.Abs(ax) <= Math.Abs(az)) hx = 1;
        else if (Math.Abs(ay) <= Math.Abs(az)) hy = 1;
        else hz = 1;
        double tx = hy * az - hz * ay, ty = hz * ax - hx * az, tz = hx * ay - hy * ax;
        double tl = Math.Sqrt(tx * tx + ty * ty + tz * tz);
        tx /= tl; ty /= tl; tz /= tl;
        double bx = ay * tz - az * ty, by = az * tx - ax * tz, bz = ax * ty - ay * tx;

        List<int> sources = [];
        List<double> uv = [];
        int[] local = new int[faces.Count * 3];
        for (int i = 0; i < faces.Count; i++)
        {
            for (int k = 0; k < 3; k++)
            {
                int v = mesh.Indices[faces[i] * 3 + k];
                if (localOf[v] < 0)
                {
                    localOf[v] = sources.Count;
                    sources.Add(v);
                    double px = mesh.Vertices[v * 3], py = mesh.Vertices[v * 3 + 1], pz = mesh.Vertices[v * 3 + 2];
                    uv.Add(px * tx + py * ty + pz * tz);
                    uv.Add(px * bx + py * by + pz * bz);
                }
                local[i * 3 + k] = localOf[v];
            }
        }
        foreach (int v in sources) localOf[v] = -1;

        double[] uvArray = [.. uv];
        double area2d = 0;
        for (int i = 0; i < faces.Count; i++)
        {
            int a = local[i * 3] * 2, b = local[i * 3 + 1] * 2, c = local[i * 3 + 2] * 2;
            area2d += 0.5 * Math.Abs((uvArray[b] - uvArray[a]) * (uvArray[c + 1] - uvArray[a + 1])
                - (uvArray[c] - uvArray[a]) * (uvArray[b + 1] - uvArray[a + 1]));
        }
        return new UvChart([.. faces], [.. sources], uvArray, local, area3d, area2d);
    }

    private static UvUnwrapResult Assemble(Mesh mesh, List<UvChart> charts)
    {
        int total = 0;
        foreach (UvChart chart in charts) total += chart.SourceVertices.Length;
        float[] vertices = new float[total * 3];
        float[] uvs = new float[total * 2];
        int[] map = new int[total];
        int[] indices = new int[mesh.Indices.Length];
        int[] faceChart = new int[mesh.TriangleCount];
        float[] sourceNormals = mesh.Normals is { } n && n.Length == mesh.Vertices.Length
            ? n
            : MeshOps.ComputeVertexNormals(new Mesh { Vertices = mesh.Vertices, Indices = mesh.Indices }).Normals!;
        float[] normals = new float[total * 3];

        int baseIndex = 0;
        for (int c = 0; c < charts.Count; c++)
        {
            UvChart chart = charts[c];
            for (int l = 0; l < chart.SourceVertices.Length; l++)
            {
                int dst = baseIndex + l, src = chart.SourceVertices[l];
                map[dst] = src;
                for (int a = 0; a < 3; a++)
                {
                    vertices[dst * 3 + a] = mesh.Vertices[src * 3 + a];
                    normals[dst * 3 + a] = sourceNormals[src * 3 + a];
                }
                uvs[dst * 2] = (float)Math.Clamp(chart.Uv[l * 2], 0.0, 1.0);
                uvs[dst * 2 + 1] = (float)Math.Clamp(chart.Uv[l * 2 + 1], 0.0, 1.0);
            }
            for (int i = 0; i < chart.Faces.Length; i++)
            {
                int face = chart.Faces[i];
                faceChart[face] = c;
                for (int k = 0; k < 3; k++)
                    indices[face * 3 + k] = baseIndex + chart.LocalTriangles[i * 3 + k];
            }
            baseIndex += chart.SourceVertices.Length;
        }
        Mesh result = new() { Vertices = vertices, Indices = indices, Normals = normals, Uvs = uvs };
        return new UvUnwrapResult(result, map, faceChart, charts.Count);
    }

    /// <summary>Per-face normals and areas plus edge adjacency, with the cone-limited region growing.</summary>
    private sealed class FaceData
    {
        private readonly int[] _neighbor;
        private readonly int[] _inSet;
        private readonly int[] _taken;
        private int _token;

        public FaceData(Mesh mesh)
        {
            int faces = mesh.TriangleCount;
            Normal = new double[faces * 3];
            Area = new double[faces];
            for (int f = 0; f < faces; f++)
            {
                int a = mesh.Indices[f * 3] * 3, b = mesh.Indices[f * 3 + 1] * 3, c = mesh.Indices[f * 3 + 2] * 3;
                double ux = mesh.Vertices[b] - (double)mesh.Vertices[a], uy = mesh.Vertices[b + 1] - (double)mesh.Vertices[a + 1];
                double uz = mesh.Vertices[b + 2] - (double)mesh.Vertices[a + 2];
                double wx = mesh.Vertices[c] - (double)mesh.Vertices[a], wy = mesh.Vertices[c + 1] - (double)mesh.Vertices[a + 1];
                double wz = mesh.Vertices[c + 2] - (double)mesh.Vertices[a + 2];
                double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
                double len = Math.Sqrt(nx * nx + ny * ny + nz * nz);
                Area[f] = 0.5 * len;
                if (len > 1e-300)
                {
                    Normal[f * 3] = nx / len; Normal[f * 3 + 1] = ny / len; Normal[f * 3 + 2] = nz / len;
                }
                else
                {
                    Normal[f * 3 + 2] = 1;
                }
            }

            _neighbor = new int[faces * 3];
            Array.Fill(_neighbor, -1);
            EdgeTable table = EdgeTable.Build(mesh.Indices, mesh.VertexCount);
            for (int s = 0; s < table.Count;)
            {
                int e = table.RunEnd(s);
                if (e - s == 2)
                {
                    int h0 = table.HalfEdges[s], h1 = table.HalfEdges[s + 1];
                    _neighbor[h0] = h1 / 3;
                    _neighbor[h1] = h0 / 3;
                }
                s = e;
            }
            _inSet = new int[faces];
            _taken = new int[faces];
        }

        public double[] Normal { get; }

        public double[] Area { get; }

        /// <summary>Splits <paramref name="faces"/> into edge-connected regions, each within the cone of its (largest-first) seed face.</summary>
        public List<List<int>> Segment(List<int> faces, double cosThreshold)
        {
            int token = ++_token;
            foreach (int f in faces) _inSet[f] = token;
            int[] order = [.. faces];
            double[] keys = new double[order.Length];
            for (int i = 0; i < order.Length; i++) keys[i] = -Area[order[i]];
            Array.Sort(keys, order);

            List<List<int>> regions = [];
            Stack<int> stack = new();
            foreach (int seed in order)
            {
                if (_taken[seed] == token) continue;
                List<int> region = [];
                double sx = Normal[seed * 3], sy = Normal[seed * 3 + 1], sz = Normal[seed * 3 + 2];
                _taken[seed] = token;
                stack.Push(seed);
                while (stack.Count > 0)
                {
                    int f = stack.Pop();
                    region.Add(f);
                    for (int c = 0; c < 3; c++)
                    {
                        int g = _neighbor[f * 3 + c];
                        if (g < 0 || _inSet[g] != token || _taken[g] == token) continue;
                        double dot = Normal[g * 3] * sx + Normal[g * 3 + 1] * sy + Normal[g * 3 + 2] * sz;
                        if (dot < cosThreshold) continue;
                        _taken[g] = token;
                        stack.Push(g);
                    }
                }
                region.Sort();
                regions.Add(region);
            }
            return regions;
        }
    }
}
