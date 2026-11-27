using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: boundary tracing on closed, holed, pinched and non-manifold surfaces.</summary>
public sealed class MeshBoundaryTests
{
    // Vertices +x, -x, +y, -y, +z, -z; faces wound outward.
    private static Mesh Octahedron() => new()
    {
        Vertices = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1],
        Indices = [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5],
    };

    private static Mesh WithoutFaces(Mesh mesh, params int[] faces)
    {
        List<int> kept = [];
        for (int f = 0; f < mesh.TriangleCount; f++)
            if (!faces.Contains(f)) kept.AddRange([mesh.Indices[f * 3], mesh.Indices[f * 3 + 1], mesh.Indices[f * 3 + 2]]);
        return new Mesh { Vertices = mesh.Vertices, Indices = [.. kept] };
    }

    [Fact]
    public void Trace_ClosedSurfaceHasNoLoops()
    {
        Assert.Empty(MeshBoundary.Trace(SyntheticMeshes.Icosphere(2)));
    }

    [Fact]
    public void Trace_OneMissingFaceIsOneClosedTriangleLoop()
    {
        IReadOnlyList<BoundaryLoop> loops = MeshBoundary.Trace(WithoutFaces(Octahedron(), 0));

        BoundaryLoop loop = Assert.Single(loops);
        Assert.True(loop.Closed);
        Assert.Equal(3, loop.EdgeCount);
        Assert.Equal(3 * Math.Sqrt(2), loop.Length, 6);
        Assert.Equal([0, 2, 4], loop.Vertices.Order());
    }

    [Fact]
    public void Trace_HolesThatShareOnlyAVertexPassThatVertexTwiceAndSplitIntoTheirTwoTriangles()
    {
        // Faces 0 (+x,+y,+z) and 7 (+x,-y,-z) are opposite: removing both leaves two triangle holes pinched at +x. The two faces left at +x are
        // two fans, so the boundary of the surface is one walk that passes +x twice.
        Mesh pinched = WithoutFaces(Octahedron(), 0, 7);

        BoundaryLoop walk = Assert.Single(MeshBoundary.Trace(pinched));

        Assert.True(walk.Closed);
        Assert.Equal(6, walk.EdgeCount);
        Assert.Equal(2, walk.Vertices.Count(v => v == 0));
        IReadOnlyList<int[]> lobes = walk.SimpleCycles();
        Assert.Equal(2, lobes.Count);
        Assert.All(lobes, c => Assert.Equal(3, c.Length));
        Assert.All(lobes, c => Assert.Contains(0, c));
        Assert.Equal([0, 2, 3, 4, 5], lobes.SelectMany(c => c).Distinct().Order());
    }

    [Fact]
    public void Trace_RingOfAHoleInASphereHasOneEdgePerNeighbour()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        List<int> around = [];
        for (int f = 0; f < sphere.TriangleCount; f++)
            if (sphere.Indices[f * 3] == 0 || sphere.Indices[f * 3 + 1] == 0 || sphere.Indices[f * 3 + 2] == 0) around.Add(f);

        IReadOnlyList<BoundaryLoop> loops = MeshBoundary.Trace(WithoutFaces(sphere, [.. around]));

        BoundaryLoop loop = Assert.Single(loops);
        Assert.True(loop.Closed);
        Assert.Equal(around.Count, loop.EdgeCount);
        Assert.Equal(loop.EdgeCount, loop.Vertices.Distinct().Count());
    }

    [Fact]
    public void Trace_EveryBoundaryEdgeIsInExactlyOneLoop()
    {
        Mesh holed = WithoutFaces(SyntheticMeshes.Icosphere(2), 3, 4, 50, 51, 200, 17);

        IReadOnlyList<BoundaryLoop> loops = MeshBoundary.Trace(holed);

        Assert.Equal(SyntheticMeshes.Topology(holed).Boundary, loops.Sum(l => l.EdgeCount));
        Assert.All(loops, l => Assert.True(l.Closed));
    }

    // Three triangles share edge 0-1 (a non-manifold edge); every other edge is a boundary edge, two per triangle, so three open chains of two edges.
    private static readonly float[] FinVertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1];

    [Theory]
    [InlineData(0, 1, 2, 1, 0, 3, 0, 1, 4)]
    [InlineData(2, 0, 1, 1, 0, 3, 0, 1, 4)]
    [InlineData(1, 2, 0, 4, 0, 1, 3, 1, 0)]
    [InlineData(0, 1, 4, 2, 0, 1, 1, 0, 3)]
    public void Trace_ANonManifoldEdgeEndsEachChainWholeWhateverTheFaceOrderAndWinding(int a, int b, int c, int d, int e, int f, int g, int h, int i)
    {
        Mesh fins = new() { Vertices = FinVertices, Indices = [a, b, c, d, e, f, g, h, i] };

        IReadOnlyList<BoundaryLoop> loops = MeshBoundary.Trace(fins);

        Assert.Equal(3, loops.Count);
        Assert.All(loops, l => Assert.False(l.Closed));
        Assert.All(loops, l => Assert.Equal(2, l.EdgeCount));
        Assert.All(loops, l => Assert.Equal(3, l.Vertices.Length));
    }

    [Fact]
    public void Trace_ARoundHoleWalkedFromAnyStartGivesOneLoop()
    {
        // The same sphere hole with its faces in reverse order starts the walk at a different edge and must give the same single loop.
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        List<int> around = [];
        for (int f = 0; f < sphere.TriangleCount; f++)
            if (sphere.Indices[f * 3] == 0 || sphere.Indices[f * 3 + 1] == 0 || sphere.Indices[f * 3 + 2] == 0) around.Add(f);
        Mesh holed = WithoutFaces(sphere, [.. around]);
        int[] reversed = new int[holed.Indices.Length];
        for (int f = 0; f < holed.TriangleCount; f++)
            Array.Copy(holed.Indices, f * 3, reversed, (holed.TriangleCount - 1 - f) * 3, 3);

        BoundaryLoop loop = Assert.Single(MeshBoundary.Trace(new Mesh { Vertices = holed.Vertices, Indices = reversed }));

        Assert.True(loop.Closed);
        Assert.Equal(around.Count, loop.EdgeCount);
    }

    [Fact]
    public void Trace_IgnoresAFaceThatRepeatsAVertex()
    {
        Mesh holed = WithoutFaces(Octahedron(), 0);
        Mesh withDegenerate = new() { Vertices = holed.Vertices, Indices = [.. holed.Indices, 0, 0, 2] };

        BoundaryLoop loop = Assert.Single(MeshBoundary.Trace(withDegenerate));

        Assert.True(loop.Closed);
        Assert.Equal(3, loop.EdgeCount);
    }
}
