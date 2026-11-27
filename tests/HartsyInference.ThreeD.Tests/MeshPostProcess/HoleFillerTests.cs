using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: width-limited hole filling on slits, round openings and pinched loops.</summary>
public sealed class HoleFillerTests
{
    private static Mesh WithoutFaces(Mesh mesh, Func<int, bool> drop)
    {
        List<int> kept = [];
        for (int f = 0; f < mesh.TriangleCount; f++)
            if (!drop(f)) kept.AddRange([mesh.Indices[f * 3], mesh.Indices[f * 3 + 1], mesh.Indices[f * 3 + 2]]);
        return new Mesh { Vertices = mesh.Vertices, Indices = [.. kept] };
    }

    private static float CentroidY(Mesh mesh, int face) =>
        (mesh.Vertices[mesh.Indices[face * 3] * 3 + 1] + mesh.Vertices[mesh.Indices[face * 3 + 1] * 3 + 1] + mesh.Vertices[mesh.Indices[face * 3 + 2] * 3 + 1]) / 3;

    // A flat grid with the cells x in [10, 50), y in [29, 31) cut out: a slit 40 cells long and 2 wide, far from the border.
    private static Mesh SlitGrid(out int cells)
    {
        cells = 60;
        Mesh grid = SyntheticMeshes.Grid(cells);
        int n = cells;
        return WithoutFaces(grid, f =>
        {
            int cell = f / 2, x = cell % n, y = cell / n;
            return x >= 10 && x < 50 && y >= 29 && y < 31;
        });
    }

    [Fact]
    public void Fill_ClosesALongThinSlitThatThePerimeterLimitLeavesOpen()
    {
        Mesh slit = SlitGrid(out int n);
        int outer = 4 * n;
        double perimeter = 2 * (40 + 2) / (double)n;
        Assert.Equal(outer + 84, SyntheticMeshes.Topology(slit).Boundary);

        Mesh byPerimeter = MeshCleaner.FillHoles(slit, perimeter * 0.5);
        HoleFillResult byWidth = HoleFiller.Fill(slit, 0.1);

        Assert.Equal(outer + 84, SyntheticMeshes.Topology(byPerimeter).Boundary);
        Assert.Equal(outer, SyntheticMeshes.Topology(byWidth.Mesh).Boundary);
        Assert.Equal(0, SyntheticMeshes.Topology(byWidth.Mesh).NonManifold);
        Assert.True(SyntheticMeshes.Topology(byWidth.Mesh).ConsistentWinding);
        Assert.Equal((1, 1, 0), (byWidth.Filled, byWidth.TooWide, byWidth.Unclosable));
        Assert.Equal(byWidth.Mesh.TriangleCount - slit.TriangleCount, byWidth.TrianglesAdded);
    }

    [Fact]
    public void Fill_LeavesASlitWiderThanTheLimitOpen()
    {
        Mesh slit = SlitGrid(out int n);

        HoleFillResult result = HoleFiller.Fill(slit, 0.5 / n);

        Assert.Same(slit, result.Mesh);
        Assert.Equal((0, 2, 0), (result.Filled, result.TooWide, result.Unclosable));
    }

    [Fact]
    public void Fill_KeepsAWideRoundOpeningOpenUntilTheLimitAllowsIt()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3);
        Mesh capped = WithoutFaces(sphere, f => CentroidY(sphere, f) > 0.3f);

        HoleFillResult tight = HoleFiller.Fill(capped, 0.3);
        HoleFillResult loose = HoleFiller.Fill(capped, 3);

        Assert.Same(capped, tight.Mesh);
        Assert.Equal(1, tight.TooWide);
        Assert.Equal(1, loose.Filled);
        Assert.True(SyntheticMeshes.IsClosedManifold(loose.Mesh));
        Assert.Equal(2, SyntheticMeshes.EulerCharacteristic(loose.Mesh));
        Assert.True(SyntheticMeshes.Topology(loose.Mesh).ConsistentWinding);
        Assert.True(SyntheticMeshes.SignedVolume(loose.Mesh) > 0);
    }

    [Fact]
    public void Fill_AddsNoVertexAndIsDeterministic()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        Mesh capped = WithoutFaces(sphere, f => CentroidY(sphere, f) > 0.5f);

        HoleFillResult first = HoleFiller.Fill(capped, 3);
        HoleFillResult second = HoleFiller.Fill(capped, 3);

        Assert.Equal(capped.VertexCount, first.Mesh.VertexCount);
        Assert.Equal(first.Mesh.Indices, second.Mesh.Indices);
    }

    [Fact]
    public void Fill_ClosesTheTwoHolesOfAPinch()
    {
        // Opposite faces of an octahedron removed: two triangle holes that meet at one vertex.
        Mesh octahedron = new()
        {
            Vertices = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1],
            Indices = [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5],
        };
        Mesh pinched = WithoutFaces(octahedron, f => f == 0 || f == 7);

        HoleFillResult result = HoleFiller.Fill(pinched, 10);

        Assert.Equal((2, 2, 0), (result.Filled, result.TrianglesAdded, result.TooWide));
        Assert.True(SyntheticMeshes.IsClosedManifold(result.Mesh));
        Assert.True(SyntheticMeshes.Topology(result.Mesh).ConsistentWinding);
        Assert.True(SyntheticMeshes.SignedVolume(result.Mesh) > 0);
    }

    [Fact]
    public void Fill_DoesNotCapALoneTriangleWithItself()
    {
        Mesh fin = new() { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2] };

        HoleFillResult result = HoleFiller.Fill(fin, 10);

        Assert.Same(fin, result.Mesh);
        Assert.Equal(1, result.Unclosable);
    }

    [Fact]
    public void Fill_LeavesAWalkThatEndsAtANonManifoldEdgeAlone()
    {
        Mesh fins = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1],
            Indices = [0, 1, 2, 1, 0, 3, 0, 1, 4],
        };

        HoleFillResult result = HoleFiller.Fill(fins, 10);

        Assert.True(result.OpenChains > 0);
        Assert.Equal(0, result.Filled);
    }

    [Fact]
    public void Clean_ClosesASlitThroughTheWidthOptionAndKeepsTheDefaultUnchanged()
    {
        Mesh slit = SlitGrid(out int n);
        MeshCleanOptions perimeterOnly = new() { SmallComponentAreaRatio = 0, UnifyOrientation = false };
        MeshCleanOptions withWidth = perimeterOnly with { HoleFillMaxWidth = 0.1 };

        Mesh untouched = MeshCleaner.Clean(slit, perimeterOnly);
        Mesh closed = MeshCleaner.Clean(slit, withWidth);

        Assert.Equal(4 * n + 84, SyntheticMeshes.Topology(untouched).Boundary);
        Assert.Equal(4 * n, SyntheticMeshes.Topology(closed).Boundary);
    }

    [Fact]
    public void Clean_ClosesACappedSphereWithOutwardWinding()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        Mesh capped = WithoutFaces(sphere, f => CentroidY(sphere, f) > 0.5f);

        Mesh closed = MeshCleaner.Clean(capped, new MeshCleanOptions { HoleFillMaxPerimeter = 0, HoleFillMaxWidth = 3 });

        Assert.True(SyntheticMeshes.IsClosedManifold(closed));
        Assert.True(SyntheticMeshes.SignedVolume(closed) > 0);
    }

    [Fact]
    public void Clean_ClosesACappedSphereWhoseFacesAreWoundBothWaysWithOneConsistentWinding()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        Mesh capped = WithoutFaces(sphere, f => CentroidY(sphere, f) > 0.5f);
        int[] mixed = (int[])capped.Indices.Clone();
        for (int f = 0; f < mixed.Length / 3; f += 3) (mixed[f * 3 + 1], mixed[f * 3 + 2]) = (mixed[f * 3 + 2], mixed[f * 3 + 1]);
        Mesh inconsistent = new() { Vertices = capped.Vertices, Indices = mixed };
        Assert.False(SyntheticMeshes.Topology(inconsistent).ConsistentWinding);

        Mesh closed = MeshCleaner.Clean(inconsistent, new MeshCleanOptions { HoleFillMaxPerimeter = 0, HoleFillMaxWidth = 3 });

        Assert.True(SyntheticMeshes.IsClosedManifold(closed));
        Assert.True(SyntheticMeshes.Topology(closed).ConsistentWinding);
        Assert.True(SyntheticMeshes.SignedVolume(closed) > 0);
    }

    [Fact]
    public void Fill_RejectsANegativeWidth()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => HoleFiller.Fill(SyntheticMeshes.Icosphere(1), -1));
    }
}
