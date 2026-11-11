using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: UV atlas validity (range, overlap, coverage, vertex map) on closed and curved shapes.</summary>
public sealed class UvUnwrapperTests
{
    public static IEnumerable<object[]> Shapes()
    {
        yield return [SyntheticMeshes.Cube(4)];
        yield return [SyntheticMeshes.Icosphere(2)];
        yield return [SyntheticMeshes.Torus(24, 12)];
        yield return [SyntheticMeshes.Grid(6)];
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Atlas_StaysInsideTheUnitSquareAndNeverOverlaps(Mesh mesh)
    {
        UvUnwrapResult result = UvUnwrapper.Unwrap(mesh);

        Assert.NotNull(result.Mesh.Uvs);
        foreach (float uv in result.Mesh.Uvs!)
            Assert.InRange(uv, 0f, 1f);
        Assert.Equal(0, CountOverlappingPairs(result.Mesh));
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void VertexMap_PointsBackAtIdenticalSourcePositionsAndFacesKeepTheirOrder(Mesh mesh)
    {
        UvUnwrapResult result = UvUnwrapper.Unwrap(mesh);

        Assert.Equal(mesh.TriangleCount, result.Mesh.TriangleCount);
        Assert.Equal(result.Mesh.VertexCount, result.VertexMap.Length);
        for (int v = 0; v < result.Mesh.VertexCount; v++)
        {
            Assert.InRange(result.VertexMap[v], 0, mesh.VertexCount - 1);
            for (int a = 0; a < 3; a++)
                Assert.Equal(mesh.Vertices[result.VertexMap[v] * 3 + a], result.Mesh.Vertices[v * 3 + a]);
        }
        for (int i = 0; i < mesh.Indices.Length; i++)
            Assert.Equal(mesh.Indices[i], result.VertexMap[result.Mesh.Indices[i]]);
        Assert.NotNull(result.Mesh.Normals);
        Assert.Equal(result.Mesh.VertexCount * 3, result.Mesh.Normals!.Length);
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Atlas_CoversAUsefulShareOfTheSquareAtUniformDensity(Mesh mesh)
    {
        UvUnwrapResult result = UvUnwrapper.Unwrap(mesh);

        double uvArea = 0, surface = 0;
        List<double> density = [];
        for (int f = 0; f < mesh.TriangleCount; f++)
        {
            double a2 = UvArea(result.Mesh, f), a3 = SurfaceArea(mesh, f);
            uvArea += a2;
            surface += a3;
            if (a3 > 0 && a2 > 0) density.Add(a2 / a3);
        }
        Assert.InRange(uvArea, 0.25, 1.0);
        // Isotropic chart scaling makes UV area track surface area chart by chart, so the global ratio is the density.
        double global = uvArea / surface;
        double median = density.Order().ElementAt(density.Count / 2);
        Assert.InRange(median / global, 0.05, 20.0);
    }

    [Fact]
    public void Cube_SplitsIntoItsSixSides()
    {
        UvUnwrapResult result = UvUnwrapper.Unwrap(SyntheticMeshes.Cube(3));

        Assert.Equal(6, result.ChartCount);
    }

    [Fact]
    public void Grid_IsASingleChartWithNoSeams()
    {
        Mesh grid = SyntheticMeshes.Grid(8);

        UvUnwrapResult result = UvUnwrapper.Unwrap(grid);

        Assert.Equal(1, result.ChartCount);
        Assert.Equal(grid.VertexCount, result.Mesh.VertexCount);
    }

    [Fact]
    public void SelfOverlappingProjection_IsSplitIntoValidCharts()
    {
        // Two turns of a gently sloped helical ramp: every normal is within ~12 degrees of +z, so one cone chart would project onto
        // itself. The overlap check has to catch that and re-split.
        const int segments = 96, turns = 2, across = 3;
        float[] verts = new float[(segments * turns + 1) * (across + 1) * 3];
        for (int s = 0; s <= segments * turns; s++)
            for (int w = 0; w <= across; w++)
            {
                double theta = 2 * Math.PI * s / segments, radius = 1.0 + 0.5 * w / across;
                int o = (s * (across + 1) + w) * 3;
                verts[o] = (float)(radius * Math.Cos(theta)); verts[o + 1] = (float)(radius * Math.Sin(theta)); verts[o + 2] = (float)(0.2 * theta);
            }
        List<int> idx = [];
        for (int s = 0; s < segments * turns; s++)
            for (int w = 0; w < across; w++)
            {
                int a = s * (across + 1) + w, b = (s + 1) * (across + 1) + w, c = b + 1, d = a + 1;
                idx.AddRange([a, b, c, a, c, d]);
            }
        Mesh ramp = new() { Vertices = verts, Indices = [.. idx] };

        UvUnwrapResult result = UvUnwrapper.Unwrap(ramp);

        Assert.True(result.ChartCount > 1);
        Assert.Equal(0, CountOverlappingPairs(result.Mesh));
        foreach (float uv in result.Mesh.Uvs!) Assert.InRange(uv, 0f, 1f);
    }

    [Fact]
    public void Unwrap_IsDeterministic()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);

        UvUnwrapResult a = UvUnwrapper.Unwrap(sphere);
        UvUnwrapResult b = UvUnwrapper.Unwrap(sphere);

        Assert.Equal(a.Mesh.Uvs, b.Mesh.Uvs);
        Assert.Equal(a.VertexMap, b.VertexMap);
        Assert.Equal(a.Mesh.Indices, b.Mesh.Indices);
    }

    [Fact]
    public void SourceNormalsAreCarriedAcrossSeams()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        MeshOps.ComputeVertexNormals(sphere);

        UvUnwrapResult result = UvUnwrapper.Unwrap(sphere);

        for (int v = 0; v < result.Mesh.VertexCount; v++)
            for (int a = 0; a < 3; a++)
                Assert.Equal(sphere.Normals![result.VertexMap[v] * 3 + a], result.Mesh.Normals![v * 3 + a]);
    }

    [Fact]
    public void RejectsInvalidOptions()
    {
        Mesh grid = SyntheticMeshes.Grid(2);
        Assert.Throws<ArgumentOutOfRangeException>(() => UvUnwrapper.Unwrap(grid, new UvUnwrapOptions { ConeHalfAngleDegrees = 95 }));
        Assert.Throws<ArgumentException>(() => UvUnwrapper.Unwrap(new Mesh { Vertices = [], Indices = [] }));
    }

    // Independent of the production check: Sutherland-Hodgman clips one triangle by the other and measures the overlap area.
    private static int CountOverlappingPairs(Mesh mesh)
    {
        float[] uv = mesh.Uvs!;
        int faces = mesh.TriangleCount;
        double[][] tris = new double[faces][];
        double[][] boxes = new double[faces][];
        for (int f = 0; f < faces; f++)
        {
            tris[f] = new double[6];
            for (int k = 0; k < 3; k++)
            {
                tris[f][k * 2] = uv[mesh.Indices[f * 3 + k] * 2];
                tris[f][k * 2 + 1] = uv[mesh.Indices[f * 3 + k] * 2 + 1];
            }
            boxes[f] = [Math.Min(tris[f][0], Math.Min(tris[f][2], tris[f][4])), Math.Min(tris[f][1], Math.Min(tris[f][3], tris[f][5])),
                Math.Max(tris[f][0], Math.Max(tris[f][2], tris[f][4])), Math.Max(tris[f][1], Math.Max(tris[f][3], tris[f][5]))];
        }
        int overlaps = 0;
        for (int a = 0; a < faces; a++)
            for (int b = a + 1; b < faces; b++)
            {
                if (boxes[a][2] <= boxes[b][0] || boxes[b][2] <= boxes[a][0] || boxes[a][3] <= boxes[b][1] || boxes[b][3] <= boxes[a][1])
                    continue;
                if (ClippedArea(tris[a], tris[b]) > 1e-9)
                    overlaps++;
            }
        return overlaps;
    }

    private static double ClippedArea(double[] subject, double[] clip)
    {
        List<(double X, double Y)> poly = [(subject[0], subject[1]), (subject[2], subject[3]), (subject[4], subject[5])];
        double orient = Cross(clip[0], clip[1], clip[2], clip[3], clip[4], clip[5]);
        for (int e = 0; e < 3 && poly.Count > 0; e++)
        {
            double ax = clip[e * 2], ay = clip[e * 2 + 1], bx = clip[(e + 1) % 3 * 2], by = clip[(e + 1) % 3 * 2 + 1];
            List<(double X, double Y)> next = [];
            for (int i = 0; i < poly.Count; i++)
            {
                (double X, double Y) cur = poly[i], prev = poly[(i + poly.Count - 1) % poly.Count];
                double sc = Side(ax, ay, bx, by, cur.X, cur.Y) * Math.Sign(orient), sp = Side(ax, ay, bx, by, prev.X, prev.Y) * Math.Sign(orient);
                if (sc >= 0)
                {
                    if (sp < 0) next.Add(Intersect(prev, cur, sp, sc));
                    next.Add(cur);
                }
                else if (sp >= 0)
                {
                    next.Add(Intersect(prev, cur, sp, sc));
                }
            }
            poly = next;
        }
        double area = 0;
        for (int i = 0; i < poly.Count; i++)
        {
            (double X, double Y) p = poly[i], q = poly[(i + 1) % poly.Count];
            area += p.X * q.Y - q.X * p.Y;
        }
        return Math.Abs(area) * 0.5;
    }

    private static (double X, double Y) Intersect((double X, double Y) p, (double X, double Y) q, double sp, double sq)
    {
        double t = sp / (sp - sq);
        return (p.X + (q.X - p.X) * t, p.Y + (q.Y - p.Y) * t);
    }

    private static double Side(double ax, double ay, double bx, double by, double px, double py) => (bx - ax) * (py - ay) - (by - ay) * (px - ax);

    private static double Cross(double ax, double ay, double bx, double by, double cx, double cy) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

    private static double UvArea(Mesh mesh, int f)
    {
        float[] uv = mesh.Uvs!;
        int a = mesh.Indices[f * 3] * 2, b = mesh.Indices[f * 3 + 1] * 2, c = mesh.Indices[f * 3 + 2] * 2;
        return 0.5 * Math.Abs((uv[b] - uv[a]) * (uv[c + 1] - uv[a + 1]) - (uv[c] - uv[a]) * (uv[b + 1] - uv[a + 1]));
    }

    private static double SurfaceArea(Mesh mesh, int f)
    {
        float[] v = mesh.Vertices;
        int a = mesh.Indices[f * 3] * 3, b = mesh.Indices[f * 3 + 1] * 3, c = mesh.Indices[f * 3 + 2] * 3;
        double ux = v[b] - v[a], uy = v[b + 1] - v[a + 1], uz = v[b + 2] - v[a + 2];
        double wx = v[c] - v[a], wy = v[c + 1] - v[a + 1], wz = v[c + 2] - v[a + 2];
        double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        return 0.5 * Math.Sqrt(nx * nx + ny * ny + nz * nz);
    }
}
