using System.Diagnostics;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: quadric decimation keeps size, shape and topology.</summary>
public sealed class MeshDecimatorTests
{
    [Fact]
    public void Icosphere_ReachesTargetAndStaysClosedManifoldWithItsBounds()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4);

        Mesh reduced = MeshDecimator.Decimate(sphere, 1000);

        Assert.InRange(reduced.TriangleCount, 998, 1000);
        Assert.True(SyntheticMeshes.IsClosedManifold(reduced));
        Assert.Equal(2, SyntheticMeshes.EulerCharacteristic(reduced));
        Assert.True(SyntheticMeshes.SignedVolume(reduced) > 0);
        AssertBoundsClose(sphere, reduced, 0.03);
    }

    [Fact]
    public void Torus_KeepsItsGenus()
    {
        Mesh torus = SyntheticMeshes.Torus(60, 30);

        Mesh reduced = MeshDecimator.Decimate(torus, 600);

        Assert.InRange(reduced.TriangleCount, 598, 600);
        Assert.True(SyntheticMeshes.IsClosedManifold(reduced));
        Assert.Equal(0, SyntheticMeshes.EulerCharacteristic(reduced));
        AssertBoundsClose(torus, reduced, 0.05);
    }

    [Fact]
    public void Decimate_IsDeterministic()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(4);

        Mesh a = MeshDecimator.Decimate(sphere, 700);
        Mesh b = MeshDecimator.Decimate(sphere, 700);

        Assert.Equal(a.Indices, b.Indices);
        Assert.Equal(a.Vertices, b.Vertices);
    }

    [Fact]
    public void OpenGrid_KeepsItsBoundaryOnTheOriginalSquare()
    {
        Mesh grid = SyntheticMeshes.Grid(20);

        Mesh reduced = MeshDecimator.Decimate(grid, 60);

        Assert.InRange(reduced.TriangleCount, 55, 60);
        (int _, int boundary, int nonManifold, bool consistent) = SyntheticMeshes.Topology(reduced);
        Assert.True(boundary > 0);
        Assert.Equal(0, nonManifold);
        Assert.True(consistent);
        (double[] min, double[] max) = SyntheticMeshes.Bounds(reduced);
        Assert.Equal(0.0, min[0], 6); Assert.Equal(0.0, min[1], 6); Assert.Equal(1.0, max[0], 6); Assert.Equal(1.0, max[1], 6);
        for (int i = 0; i < reduced.VertexCount; i++)
            Assert.Equal(0.0, reduced.Vertices[i * 3 + 2], 6);
    }

    [Fact]
    public void TargetAtOrAboveFaceCount_ReturnsACopy()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);

        Mesh same = MeshDecimator.Decimate(sphere, sphere.TriangleCount + 10);

        Assert.Equal(sphere.Indices, same.Indices);
        Assert.NotSame(sphere.Indices, same.Indices);
    }

    [Fact]
    public void DecimateStaged_RunsTwoPassesAndLeavesAClosedOutwardMesh()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(5);

        Mesh reduced = MeshDecimator.DecimateStaged(sphere, 500);

        Assert.InRange(reduced.TriangleCount, 450, 520);
        Assert.True(SyntheticMeshes.IsClosedManifold(reduced));
        Assert.True(SyntheticMeshes.SignedVolume(reduced) > 0);
        AssertBoundsClose(sphere, reduced, 0.05);
    }

    [Fact]
    public void RejectsBadArguments()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(1);
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshDecimator.Decimate(sphere, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => MeshDecimator.DecimateStaged(sphere, 10, 0));
        Assert.Throws<ArgumentException>(() => MeshDecimator.Decimate(new Mesh { Vertices = new float[9], Indices = [0, 1, 5] }, 1));
    }

    [Fact]
    public void TwoHundredThousandFaces_DecimateWithinAGenerousBudget()
    {
        Mesh torus = SyntheticMeshes.Torus(330, 330);
        Assert.True(torus.TriangleCount >= 200_000);

        Stopwatch watch = Stopwatch.StartNew();
        Mesh reduced = MeshDecimator.Decimate(torus, 20_000);
        watch.Stop();

        Assert.InRange(reduced.TriangleCount, 19_990, 20_000);
        Assert.True(SyntheticMeshes.IsClosedManifold(reduced));
        Assert.True(watch.Elapsed < TimeSpan.FromMinutes(3), $"Decimation took {watch.Elapsed}.");
    }

    private static void AssertBoundsClose(Mesh original, Mesh reduced, double relativeTolerance)
    {
        (double[] omin, double[] omax) = SyntheticMeshes.Bounds(original);
        (double[] rmin, double[] rmax) = SyntheticMeshes.Bounds(reduced);
        double extent = 0;
        for (int a = 0; a < 3; a++) extent = Math.Max(extent, omax[a] - omin[a]);
        for (int a = 0; a < 3; a++)
        {
            Assert.InRange(Math.Abs(omin[a] - rmin[a]), 0, relativeTolerance * extent);
            Assert.InRange(Math.Abs(omax[a] - rmax[a]), 0, relativeTolerance * extent);
        }
    }
}
