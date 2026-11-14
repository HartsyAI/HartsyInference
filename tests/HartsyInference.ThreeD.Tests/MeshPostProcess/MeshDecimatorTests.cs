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
    public void Tetrahedron_IsNeverCollapsedIntoCoincidentFaces()
    {
        Mesh tetrahedron = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            Indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        };

        Mesh reduced = MeshDecimator.Decimate(tetrahedron, 2);

        Assert.Equal(4, reduced.TriangleCount);
        Assert.True(SyntheticMeshes.IsClosedManifold(reduced));
        Assert.True(SyntheticMeshes.SignedVolume(reduced) > 0);
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

    // Every fourth face removed: boundary vertices everywhere, so the link of most interior edges has two boundary ends.
    private static Mesh TornSphere()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(5, 1f);
        List<int> kept = [];
        for (int f = 0; f < sphere.TriangleCount; f++)
            if (f % 4 != 0)
                kept.AddRange(sphere.Indices[(f * 3)..(f * 3 + 3)]);
        return new Mesh { Vertices = sphere.Vertices, Indices = [.. kept] };
    }

    [Fact]
    public void BoundaryPinch_LetsATornMeshReachWhatItsBoundariesOtherwiseBlock()
    {
        Mesh torn = TornSphere();

        int pinned = MeshDecimator.Decimate(torn, 500).TriangleCount;
        int pinching = MeshDecimator.Decimate(torn, 500, new MeshDecimatorOptions { AllowBoundaryPinch = true }).TriangleCount;

        Assert.True(pinned > 1000, $"without the pinch it stopped at {pinned}");
        Assert.True(pinching < pinned / 2, $"with the pinch it reached {pinching}, without {pinned}");
    }

    [Fact]
    public void Staged_FallsBackToThePinchWhenAPassStallsAboveTheTarget()
    {
        Mesh torn = TornSphere();
        MeshCleanOptions clean = new() { SmallComponentAreaRatio = 0, HoleFillMaxPerimeter = 0, UnifyOrientation = false };

        Mesh staged = MeshDecimator.DecimateStaged(torn, 500, 3, clean);

        Assert.True(staged.TriangleCount < 1000, $"the staged result kept {staged.TriangleCount} faces");
    }
}
