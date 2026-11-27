using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: BVH closest-point queries against a brute-force reference.</summary>
public sealed class TriangleBvhTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void ClosestPoint_MatchesBruteForceOnAnIcosphere(int subdivisions)
    {
        Mesh mesh = SyntheticMeshes.Icosphere(subdivisions, 0.8f);
        AssertMatchesBruteForce(mesh, 400, 3.0);
    }

    [Fact]
    public void ClosestPoint_MatchesBruteForceOnATorus()
    {
        Mesh mesh = SyntheticMeshes.Torus(24, 12);
        AssertMatchesBruteForce(mesh, 400, 2.5);
    }

    [Fact]
    public void ClosestPoint_BarycentricsReproduceTheReturnedPoint()
    {
        Mesh mesh = SyntheticMeshes.Icosphere(2);
        TriangleBvh bvh = new(mesh);
        BvhHit hit = bvh.ClosestPoint(0.3, 2.0, -0.4);
        int a = mesh.Indices[hit.Triangle * 3] * 3, b = mesh.Indices[hit.Triangle * 3 + 1] * 3, c = mesh.Indices[hit.Triangle * 3 + 2] * 3;
        double w0 = 1 - hit.U - hit.V;
        for (int k = 0; k < 3; k++)
        {
            double p = w0 * mesh.Vertices[a + k] + hit.U * mesh.Vertices[b + k] + hit.V * mesh.Vertices[c + k];
            Assert.Equal(p, new[] { hit.X, hit.Y, hit.Z }[k], 9);
        }
        Assert.Equal(hit.Distance, bvh.Distance(0.3, 2.0, -0.4), 12);
    }

    [Theory]
    [InlineData(0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f)] // A == B
    [InlineData(0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f)] // A == C
    [InlineData(0f, 0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f)] // B == C
    public void ClosestPoint_OnACollapsedEdgeTriangleIsTheClosestPointOfItsSegment(
        float ax, float ay, float az, float bx, float by, float bz, float cx, float cy, float cz)
    {
        Mesh mesh = new() { Vertices = [ax, ay, az, bx, by, bz, cx, cy, cz], Indices = [0, 1, 2] };
        TriangleBvh bvh = new(mesh);

        BvhHit hit = bvh.ClosestPoint(0.5, 1.0, 0.0);

        Assert.Equal(0, hit.Triangle);
        Assert.Equal(0.5, hit.X, 9);
        Assert.Equal(0.0, hit.Y, 9);
        Assert.Equal(0.0, hit.Z, 9);
        Assert.Equal(1.0, hit.Distance, 9);
    }

    [Fact]
    public void ClosestPoint_OnTheSurfaceIsZeroDistance()
    {
        Mesh mesh = SyntheticMeshes.Grid(4);
        TriangleBvh bvh = new(mesh);
        Assert.Equal(0.0, bvh.Distance(0.37, 0.61, 0.0), 12);
        Assert.Equal(0.5, bvh.Distance(0.37, 0.61, 0.5), 12);
    }

    private static void AssertMatchesBruteForce(Mesh mesh, int queries, double spread)
    {
        TriangleBvh bvh = new(mesh);
        Random rng = new(1234);
        for (int q = 0; q < queries; q++)
        {
            double x = (rng.NextDouble() * 2 - 1) * spread, y = (rng.NextDouble() * 2 - 1) * spread, z = (rng.NextDouble() * 2 - 1) * spread;
            double best = double.MaxValue;
            for (int t = 0; t < mesh.TriangleCount; t++)
                best = Math.Min(best, BruteDistanceSquared(mesh, t, x, y, z));
            BvhHit hit = bvh.ClosestPoint(x, y, z);
            Assert.Equal(best, hit.DistanceSquared, 9);
        }
    }

    // Independent of the production routine: project onto the plane, accept if inside, otherwise take the nearest edge point.
    private static double BruteDistanceSquared(Mesh mesh, int t, double px, double py, double pz)
    {
        double[][] v = new double[3][];
        for (int k = 0; k < 3; k++)
        {
            int i = mesh.Indices[t * 3 + k] * 3;
            v[k] = [mesh.Vertices[i], mesh.Vertices[i + 1], mesh.Vertices[i + 2]];
        }
        double[] n = Cross(Sub(v[1], v[0]), Sub(v[2], v[0]));
        double nn = Dot(n, n);
        double[] p = [px, py, pz];
        if (nn > 0)
        {
            double d = Dot(Sub(p, v[0]), n) / nn;
            double[] q = [px - n[0] * d, py - n[1] * d, pz - n[2] * d];
            bool inside = true;
            for (int k = 0; k < 3; k++)
            {
                double[] edge = Sub(v[(k + 1) % 3], v[k]);
                if (Dot(Cross(edge, Sub(q, v[k])), n) < 0) inside = false;
            }
            if (inside) return Dot(Sub(p, q), Sub(p, q));
        }
        double best = double.MaxValue;
        for (int k = 0; k < 3; k++)
        {
            double[] a = v[k], e = Sub(v[(k + 1) % 3], a);
            double ee = Dot(e, e);
            double s = ee > 0 ? Math.Clamp(Dot(Sub(p, a), e) / ee, 0, 1) : 0;
            double[] c = [a[0] + e[0] * s, a[1] + e[1] * s, a[2] + e[2] * s];
            best = Math.Min(best, Dot(Sub(p, c), Sub(p, c)));
        }
        return best;
    }

    private static double[] Sub(double[] a, double[] b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    private static double Dot(double[] a, double[] b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    private static double[] Cross(double[] a, double[] b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
