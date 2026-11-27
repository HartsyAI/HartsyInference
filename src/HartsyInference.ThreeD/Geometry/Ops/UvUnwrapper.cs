namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Chart-based UV unwrapping. Charts grow face by face from the largest unassigned face (<see cref="UvChartGrower"/>): a face joins when its
/// normal stays within a cone of the chart's projection normal and its projected triangle overlaps nothing already in the chart, so a chart is
/// valid by construction and a noisy surface costs only the facets that really fold. Small charts are then dissolved into large neighbours at a
/// wider cone, each chart is projected onto the plane of its area-weighted mean normal when that stays valid, and the charts are rotated, scaled
/// to a uniform texel density and packed into the unit square. Deterministic; the output has the same faces in the same order as the input.
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
        if (options.ConeHalfAngleDegrees is <= 0 or >= 90 || options.AbsorbConeHalfAngleDegrees is <= 0 or >= 90 || options.SteepAbsorbConeHalfAngleDegrees is <= 0 or >= 90 || options.SteepAbsorbMaxAreaRatio < 0 || options.AbsorbBelowFaces < 0
            || options.TextureSize <= 0 || options.PaddingTexels < 0 || options.MinPaddingTexels < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Cone angles must be in (0,90) and sizes must be positive.");

        FaceData data = new(mesh);
        List<UvChart> charts = BuildCharts(mesh, data, options, cancellationToken);
        double padding = UvAtlasPacker.Pack(charts, options.PaddingTexels, Math.Min(options.MinPaddingTexels, options.PaddingTexels), options.TextureSize);
        return Assemble(mesh, charts, padding);
    }

    private static List<UvChart> BuildCharts(Mesh mesh, FaceData data, UvUnwrapOptions options, CancellationToken cancellationToken)
    {
        int faces = mesh.TriangleCount;
        double coneCos = Math.Cos(options.ConeHalfAngleDegrees * Math.PI / 180.0);
        double absorbCos = Math.Cos(options.AbsorbConeHalfAngleDegrees * Math.PI / 180.0);
        double steepCos = Math.Cos(options.SteepAbsorbConeHalfAngleDegrees * Math.PI / 180.0);
        UvChartGrower grower = new(mesh, data.Normal, data.Area, data.Neighbor);

        // Seeds largest face first, lowest index on ties, so the charts do not depend on the sort algorithm.
        int[] order = new int[faces];
        double[] keys = new double[faces];
        for (int f = 0; f < faces; f++)
        {
            order[f] = f;
            keys[f] = -data.Area[f];
        }

        Array.Sort(keys, order);
        StableByKey(order, keys);

        int[] chartOf = new int[faces];
        Array.Fill(chartOf, -1);
        List<List<int>> chartFaces = [];
        List<double[]> chartNormal = [];

        void Seed(IEnumerable<int> seeds)
        {
            foreach (int seed in seeds)
            {
                if (chartOf[seed] >= 0)
                    continue;
                int id = chartFaces.Count;
                if ((id & 255) == 0)
                    cancellationToken.ThrowIfCancellationRequested();
                List<int> members = [seed];
                chartOf[seed] = id;
                double[] normal = grower.SeedNormal(seed, coneCos);
                grower.Grow(members, normal, coneCos, f => chartOf[f] < 0, f => chartOf[f] = id, cancellationToken);
                chartFaces.Add(members);
                chartNormal.Add(normal);
            }
        }

        Seed(order);

        // Dissolve the small charts: their faces return to the pool, and each large chart (in index order) takes what its projection can hold
        // at the wider cone. What no large chart can take is seeded again, and a last round lets large charts take the small, steep leftovers at
        // the steepest cone (they keep a sliver of texels and are inpainted from their neighbours, which is what a noise facet deserves).
        if (options.AbsorbBelowFaces > 0)
        {
            double meanArea = 0;
            foreach (double a in data.Area)
                meanArea += a;
            meanArea /= Math.Max(1, faces);
            (double Cos, double MaxArea)[] rounds =
            [
                (absorbCos, double.MaxValue),
                (steepCos, options.SteepAbsorbMaxAreaRatio * meanArea),
            ];
            foreach ((double cos, double maxArea) in rounds)
            {
                int first = chartFaces.Count;
                bool[] dead = new bool[first];
                bool any = false;
                for (int c = 0; c < first; c++)
                {
                    if (chartFaces[c].Count == 0 || chartFaces[c].Count >= options.AbsorbBelowFaces)
                        continue;
                    dead[c] = true;
                    any = true;
                    foreach (int f in chartFaces[c])
                        chartOf[f] = -1;
                }

                if (!any)
                    break;
                for (int c = 0; c < first; c++)
                {
                    if (dead[c] || chartFaces[c].Count == 0)
                        continue;
                    int id = c;
                    grower.Grow(chartFaces[c], chartNormal[c], cos, f => chartOf[f] < 0 && data.Area[f] <= maxArea, f => chartOf[f] = id, cancellationToken);
                }

                Seed(order.Where(f => chartOf[f] < 0).ToArray());
                for (int c = 0; c < first; c++)
                {
                    if (dead[c])
                        chartFaces[c] = [];
                }
            }
        }

        List<UvChart> charts = [];
        int[] localOf = new int[mesh.VertexCount];
        Array.Fill(localOf, -1);
        for (int c = 0; c < chartFaces.Count; c++)
        {
            if (chartFaces[c].Count == 0)
                continue;
            if ((c & 255) == 0)
                cancellationToken.ThrowIfCancellationRequested();
            List<int> members = chartFaces[c];
            members.Sort();
            charts.Add(Project(mesh, data, members, chartNormal[c], steepCos, localOf));
        }

        return charts;
    }

    // Array.Sort is not stable: put equal keys back in index order.
    private static void StableByKey(int[] order, double[] keys)
    {
        int start = 0;
        for (int i = 1; i <= order.Length; i++)
        {
            if (i < order.Length && keys[i] == keys[start])
                continue;
            if (i - start > 1)
                Array.Sort(order, start, i - start);
            start = i;
        }
    }

    // The chart on the plane of its area-weighted mean normal, which spreads the stretch evenly; when that projection folds or overlaps,
    // on the plane it was grown on, which is valid by construction.
    private static UvChart Project(Mesh mesh, FaceData data, List<int> faces, double[] grownNormal, double minCos, int[] localOf)
    {
        double ax = 0, ay = 0, az = 0;
        foreach (int f in faces)
        {
            ax += data.Normal[f * 3] * data.Area[f];
            ay += data.Normal[f * 3 + 1] * data.Area[f];
            az += data.Normal[f * 3 + 2] * data.Area[f];
        }

        double len = Math.Sqrt(ax * ax + ay * ay + az * az);
        if (faces.Count > 1 && len > 1e-300)
        {
            double mx = ax / len, my = ay / len, mz = az / len;
            // A face must keep at least the share of its area the steepest absorb round allows; past it the mean plane would squash a face a
            // few degrees from edge-on to a line (few or no texels) where the grown plane gave it more.
            bool keepsEveryFace = true;
            foreach (int f in faces)
            {
                if (data.Area[f] > 0 && data.Normal[f * 3] * mx + data.Normal[f * 3 + 1] * my + data.Normal[f * 3 + 2] * mz < minCos)
                {
                    keepsEveryFace = false;
                    break;
                }
            }

            if (keepsEveryFace)
            {
                UvChart mean = ProjectOnto(mesh, data, faces, [mx, my, mz], localOf);
                if (UvChartValidator.IsValid(mean, data.Area))
                    return mean;
            }
        }

        return ProjectOnto(mesh, data, faces, grownNormal, localOf);
    }

    private static UvChart ProjectOnto(Mesh mesh, FaceData data, List<int> faces, double[] normal, int[] localOf)
    {
        double area3d = 0;
        foreach (int f in faces)
            area3d += data.Area[f];
        UvChartGrower.Basis(normal, out double[] t, out double[] b);
        double tx = t[0], ty = t[1], tz = t[2], bx = b[0], by = b[1], bz = b[2];

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

        foreach (int v in sources)
            localOf[v] = -1;

        double[] uvArray = [.. uv];
        double area2d = 0;
        for (int i = 0; i < faces.Count; i++)
        {
            int a = local[i * 3] * 2, b2 = local[i * 3 + 1] * 2, c = local[i * 3 + 2] * 2;
            area2d += 0.5 * Math.Abs((uvArray[b2] - uvArray[a]) * (uvArray[c + 1] - uvArray[a + 1])
                - (uvArray[c] - uvArray[a]) * (uvArray[b2 + 1] - uvArray[a + 1]));
        }

        return new UvChart([.. faces], [.. sources], uvArray, local, area3d, area2d);
    }

    private static UvUnwrapResult Assemble(Mesh mesh, List<UvChart> charts, double paddingTexels)
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
        return new UvUnwrapResult(result, map, faceChart, charts.Count, paddingTexels);
    }

    /// <summary>Per-face normals and areas plus edge adjacency.</summary>
    private sealed class FaceData
    {
        private readonly int[] _neighbor;

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
        }

        public int[] Neighbor => _neighbor;

        public double[] Normal { get; }

        public double[] Area { get; }
    }
}
