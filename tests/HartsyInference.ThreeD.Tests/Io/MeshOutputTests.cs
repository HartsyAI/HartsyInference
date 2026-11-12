using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Tests.Io;

public sealed class MeshOutputTests
{
    private static Mesh Tetrahedron(float offset = 0f) => new()
    {
        Vertices = [offset, 0, 0, offset + 1, 0, 0, offset, 1, 0, offset, 0, 1],
        Indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
    };

    [Fact]
    public void AClosedSurfaceIsWatertightWithOneComponent()
    {
        MeshStats stats = MeshStatistics.Measure(Tetrahedron());

        Assert.Equal((4, 4, 1), (stats.VertexCount, stats.TriangleCount, stats.Components));
        Assert.Equal((0L, 0L), (stats.BoundaryEdges, stats.NonManifoldEdges));
        Assert.True(stats.Watertight);
        Assert.Equal([0f, 0f, 0f], stats.Min);
        Assert.Equal([1f, 1f, 1f], stats.Max);
    }

    [Fact]
    public void TwoSeparatePiecesAreTwoComponents()
    {
        Mesh a = Tetrahedron();
        Mesh b = Tetrahedron(5f);
        Mesh both = new() { Vertices = [.. a.Vertices, .. b.Vertices], Indices = [.. a.Indices, .. b.Indices.Select(i => i + 4)] };

        Assert.Equal(2, MeshStatistics.Measure(both).Components);
    }

    [Fact]
    public void ALoneTriangleHasThreeBoundaryEdges()
    {
        Mesh triangle = new() { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2] };

        MeshStats stats = MeshStatistics.Measure(triangle);

        Assert.Equal(3L, stats.BoundaryEdges);
        Assert.False(stats.Watertight);
    }

    [Fact]
    public void ASeamThatDuplicatesVerticesIsNotABoundary()
    {
        // Two triangles of a square sharing the diagonal by position but not by index (as a UV seam does).
        Mesh square = new() { Vertices = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0], Indices = [0, 1, 2, 3, 4, 5] };

        MeshStats stats = MeshStatistics.Measure(square);

        Assert.Equal(6, stats.VertexCount);
        Assert.Equal(4L, stats.BoundaryEdges);
        Assert.Equal(1, stats.Components);
    }

    [Fact]
    public void AnEdgeOfThreeFacesIsNonManifold()
    {
        Mesh fin = new() { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1], Indices = [0, 1, 2, 0, 1, 3, 0, 1, 4] };

        Assert.Equal(1L, MeshStatistics.Measure(fin).NonManifoldEdges);
    }

    [Fact]
    public void TheInspectorReadsContentsAndBoundsWithoutTheBinaryChunk()
    {
        Mesh mesh = Tetrahedron();
        MeshOps.ComputeVertexNormals(mesh);

        GlbSummary summary = GlbInspector.Read(GlbWriter.Write(mesh));

        Assert.Equal(1, summary.MeshCount);
        Assert.True(summary.HasNormals);
        Assert.False(summary.HasTexture);
        Assert.False(summary.HasSkin);
        Assert.Empty(summary.AnimationNames);
        Assert.Equal([0f, 0f, 0f], summary.Min!);
        Assert.Equal([1f, 1f, 1f], summary.Max!);
    }

    [Fact]
    public void NormalizeCentresScalesAndTurnsWithoutChangingTheVertexData()
    {
        Mesh mesh = new() { Vertices = [0, 0, 0, 2, 0, 0, 0, 1, 0], Indices = [0, 1, 2] };
        byte[] glb = GlbWriter.Write(mesh);

        byte[] framed = GlbFrame.Normalize(glb, Matrix4x4.CreateRotationY(MathF.PI));
        Mesh read = GlbReader.Parse(framed);

        // The half turn maps x to -x: the x range 0..2 becomes -2..0, is centred to -1..1 and scaled by 0.5 to -0.5..0.5, so the far vertex lands at -0.5.
        Assert.Equal(-0.5f, read.Vertices[3], 4);
        Assert.Equal(0.5f, read.Vertices[0], 4);
        Assert.Equal(1f, read.Vertices.Where((_, i) => i % 3 == 0).Max() - read.Vertices.Where((_, i) => i % 3 == 0).Min(), 4);
        Assert.Equal(glb.Length, GlbContainer.ReplaceJson(glb, GlbContainer.Split(glb).Json).Length);
        GlbSummary summary = GlbInspector.Read(framed);
        Assert.Equal(1, summary.MeshCount);
    }

    [Fact]
    public void NormalizeWithAnIdentityRotationOnlyCentresAndScales()
    {
        Mesh mesh = Tetrahedron();

        Mesh read = GlbReader.Parse(GlbFrame.Normalize(GlbWriter.Write(mesh), Matrix4x4.Identity, longestSide: 2f));

        float[] x = [.. read.Vertices.Where((_, i) => i % 3 == 0)];
        Assert.Equal(-1f, x.Min(), 4);
        Assert.Equal(1f, x.Max(), 4);
    }

    [Fact]
    public void AFileWithoutASceneCannotBeFramed()
    {
        Assert.ThrowsAny<Exception>(() => GlbFrame.Wrap([0x67, 0x6C, 0x54, 0x46], Matrix4x4.Identity));
    }
}
