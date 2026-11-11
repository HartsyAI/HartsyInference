using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>
/// Unit lane, synthetic meshes with the fault of a dual-grid mesh: a smooth surface with voxel-scale noise. The first unwrapper split a chart
/// whenever one facet folded, down to single triangles, and a TRELLIS.2 mesh of a bust came out as 111,770 charts for 1,001,330 faces. On the
/// noisy sphere below (20,480 faces) the first unwrapper made 4,647 charts (1,032 single faces, 1.44 vertices per face, 25.0% of the atlas covered)
/// and this one makes 933 (273, 0.81, 32.5%); the limits are absolute and sit between the two.
/// </summary>
public sealed class UvUnwrapperNoisyMeshTests
{
    private readonly ITestOutputHelper _output;

    public UvUnwrapperNoisyMeshTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void NoisySphere_KeepsChartsLargeAndTheAtlasUseful()
    {
        Mesh sphere = NoisySphere(5, 0.04f);

        UvUnwrapResult result = UvUnwrapper.Unwrap(sphere);

        int[] sizes = new int[result.ChartCount];
        foreach (int chart in result.FaceChart)
            sizes[chart]++;
        int singles = sizes.Count(s => s == 1);
        double uvArea = 0;
        for (int f = 0; f < sphere.TriangleCount; f++)
            uvArea += UvArea(result.Mesh, f);
        _output.WriteLine($"faces {sphere.TriangleCount}, charts {result.ChartCount}, single-face charts {singles}, vertices per face {(double)result.Mesh.VertexCount / sphere.TriangleCount:F3}, uv area {uvArea:P1}, padding {result.PaddingTexels} texels");

        Assert.True(result.ChartCount <= 1100, $"{result.ChartCount} charts for {sphere.TriangleCount} faces");
        Assert.True(singles <= 400, $"{singles} single-face charts");
        Assert.True((double)result.Mesh.VertexCount / sphere.TriangleCount <= 0.9, "most vertices should be shared inside charts");
        Assert.InRange(uvArea, 0.28, 1.0);
        Assert.Equal(4.0, result.PaddingTexels);
    }

    [Fact]
    public void NoisySphere_AtlasHasNoOverlapAndStaysInTheUnitSquare()
    {
        Mesh sphere = NoisySphere(4, 0.04f);

        UvUnwrapResult result = UvUnwrapper.Unwrap(sphere);

        Assert.All(result.Mesh.Uvs!, uv => Assert.InRange(uv, 0f, 1f));
        Assert.Equal(0, CountOverlappingPairs(result.Mesh));
    }

    [Fact]
    public void SteepFacetsAreAbsorbedIntoTheNeighbouringChart()
    {
        // A flat grid with one vertex pulled up: its six faces are steeper than the 50 degree cone but their projection onto the grid's plane is
        // valid, so they join the grid's chart instead of costing a chart of their own.
        Mesh grid = SyntheticMeshes.Grid(10);
        int centre = (grid.VertexCount - 1) / 2;
        grid.Vertices[centre * 3 + 2] += 3f * GridStep(grid);

        UvUnwrapResult absorbed = UvUnwrapper.Unwrap(grid);
        UvUnwrapResult kept = UvUnwrapper.Unwrap(grid, new UvUnwrapOptions { AbsorbBelowFaces = 0 });

        Assert.Equal(1, absorbed.ChartCount);
        Assert.True(kept.ChartCount > 1);
        Assert.Equal(0, CountOverlappingPairs(absorbed.Mesh));
    }

    [Fact]
    public void FacesOfEqualArea_SeedInIndexOrder()
    {
        // Two separate unit squares of equal area: ties must break on the face index, so the first square is chart 0 on every run.
        Mesh two = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 5, 0, 0, 6, 0, 0, 6, 1, 0, 5, 1, 0],
            Indices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
        };

        UvUnwrapResult result = UvUnwrapper.Unwrap(two);

        Assert.Equal(2, result.ChartCount);
        Assert.Equal([0, 0, 1, 1], result.FaceChart);
    }

    [Fact]
    public void ManyTinyCharts_NeverShrinkTheGutterBelowTheMinimum()
    {
        // 600 separate unit squares cannot all fit with a 4 texel gutter at a useful scale; the gutter may halve but not below the minimum.
        List<float> verts = [];
        List<int> idx = [];
        for (int q = 0; q < 600; q++)
        {
            int b = verts.Count / 3;
            float x = (q % 30) * 3f, y = (q / 30) * 3f;
            verts.AddRange([x, y, 0, x + 1, y, 0, x + 1, y + 1, 0, x, y + 1, 0]);
            idx.AddRange([b, b + 1, b + 2, b, b + 2, b + 3]);
        }

        Mesh squares = new() { Vertices = [.. verts], Indices = [.. idx] };

        UvUnwrapResult result = UvUnwrapper.Unwrap(squares, new UvUnwrapOptions { TextureSize = 256, PaddingTexels = 4, MinPaddingTexels = 1 });

        Assert.Equal(600, result.ChartCount);
        Assert.InRange(result.PaddingTexels, 1.0, 4.0);
        Assert.Equal(0, CountOverlappingPairs(result.Mesh));
    }

    [Fact]
    public void ARequestedPaddingBelowTheDefaultMinimumIsStillHonoured()
    {
        // An unpadded atlas was a valid request before the minimum existed; the minimum follows the padding down instead of throwing.
        Mesh grid = SyntheticMeshes.Grid(2);

        UvUnwrapResult none = UvUnwrapper.Unwrap(grid, new UvUnwrapOptions { PaddingTexels = 0 });
        UvUnwrapResult two = UvUnwrapper.Unwrap(grid, new UvUnwrapOptions { PaddingTexels = 2, MinPaddingTexels = 3 });

        Assert.Equal(0.0, none.PaddingTexels);
        Assert.Equal(2.0, two.PaddingTexels);
        Assert.Throws<ArgumentOutOfRangeException>(() => UvUnwrapper.Unwrap(grid, new UvUnwrapOptions { MinPaddingTexels = -1 }));
    }

    [Fact]
    public void SeedOnASharpFold_KeepsItsOwnProjection()
    {
        // The area-weighted mean of the seed and its neighbour is perpendicular to the second triangle: projected along it, that triangle would
        // have no UV area and the bake would skip it. The seed falls back to its own normal, so both triangles keep a positive UV area.
        Mesh fold = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 1],
            Indices = [0, 1, 2, 1, 0, 3],
        };

        UvUnwrapResult result = UvUnwrapper.Unwrap(fold);

        Assert.Equal(2, result.ChartCount);
        Assert.True(UvArea(result.Mesh, 0) > 1e-9);
        Assert.True(UvArea(result.Mesh, 1) > 1e-9);
    }

    [Fact]
    public void SmallSelfOverlappingComponent_NextToAVastOne_IsStillSplit()
    {
        // A helical ramp (about 0.03 per edge) next to one huge triangle (1000 across): tolerances taken from the whole mesh would call every
        // ramp triangle a sliver and skip its overlap checks, so the ramp would grow into one self-overlapping chart.
        const int segments = 96, turns = 2, across = 3;
        List<float> verts = [];
        for (int s = 0; s <= segments * turns; s++)
        {
            for (int w = 0; w <= across; w++)
            {
                double theta = 2 * Math.PI * s / segments, radius = 1.0 + 0.5 * w / across;
                verts.AddRange([(float)(radius * Math.Cos(theta)), (float)(radius * Math.Sin(theta)), (float)(0.2 * theta)]);
            }
        }

        List<int> idx = [];
        for (int s = 0; s < segments * turns; s++)
        {
            for (int w = 0; w < across; w++)
            {
                int a = s * (across + 1) + w, b = (s + 1) * (across + 1) + w, c = b + 1, d = a + 1;
                idx.AddRange([a, b, c, a, c, d]);
            }
        }

        int far = verts.Count / 3;
        verts.AddRange([5000, 5000, 0, 6000, 5000, 0, 5000, 6000, 0]);
        idx.AddRange([far, far + 1, far + 2]);
        Mesh mesh = new() { Vertices = [.. verts], Indices = [.. idx] };

        UvUnwrapResult result = UvUnwrapper.Unwrap(mesh);

        Assert.True(result.ChartCount > 2);
        Assert.Equal(0, CountOverlappingPairs(result.Mesh));
    }

    private static float GridStep(Mesh grid)
    {
        float minX = float.MaxValue, maxX = float.MinValue;
        for (int v = 0; v < grid.VertexCount; v++)
        {
            minX = Math.Min(minX, grid.Vertices[v * 3]);
            maxX = Math.Max(maxX, grid.Vertices[v * 3]);
        }

        return (maxX - minX) / 10f;
    }

    // An icosphere whose vertices move along their radius by a deterministic hash of the source position, so welded vertices stay welded.
    private static Mesh NoisySphere(int subdivisions, float amplitude)
    {
        Mesh sphere = SyntheticMeshes.Icosphere(subdivisions);
        for (int v = 0; v < sphere.VertexCount; v++)
        {
            float x = sphere.Vertices[v * 3], y = sphere.Vertices[v * 3 + 1], z = sphere.Vertices[v * 3 + 2];
            uint h = (uint)(BitConverter.SingleToInt32Bits(x) * 73856093) ^ (uint)(BitConverter.SingleToInt32Bits(y) * 19349663) ^ (uint)(BitConverter.SingleToInt32Bits(z) * 83492791);
            h ^= h >> 13;
            h *= 0x5bd1e995;
            h ^= h >> 15;
            float scale = 1f + amplitude * ((h & 0xFFFF) / 65535f * 2f - 1f);
            sphere.Vertices[v * 3] = x * scale;
            sphere.Vertices[v * 3 + 1] = y * scale;
            sphere.Vertices[v * 3 + 2] = z * scale;
        }

        return sphere;
    }

    private static double UvArea(Mesh mesh, int f)
    {
        float[] uv = mesh.Uvs!;
        int a = mesh.Indices[f * 3] * 2, b = mesh.Indices[f * 3 + 1] * 2, c = mesh.Indices[f * 3 + 2] * 2;
        return 0.5 * Math.Abs((uv[b] - uv[a]) * (uv[c + 1] - uv[a + 1]) - (uv[c] - uv[a]) * (uv[b + 1] - uv[a + 1]));
    }

    // Brute-force overlap count over bounding-box-culled pairs, by clipping one triangle with the other.
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

        // Sweep on x: sort by the box's left edge and stop comparing once the next box starts past this one's right edge.
        int[] order = [.. Enumerable.Range(0, faces).OrderBy(f => boxes[f][0]).ThenBy(f => f)];
        int overlaps = 0;
        for (int i = 0; i < faces; i++)
        {
            int a = order[i];
            for (int j = i + 1; j < faces; j++)
            {
                int b = order[j];
                if (boxes[b][0] >= boxes[a][2])
                    break;
                if (boxes[a][3] <= boxes[b][1] || boxes[b][3] <= boxes[a][1])
                    continue;
                if (ClippedArea(tris[a], tris[b]) > 1e-9)
                    overlaps++;
            }
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
                    if (sp < 0)
                        next.Add(Intersect(prev, cur, sp, sc));
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
}
