using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic meshes: the repair a shape-only TRELLIS.2 / Pixal3D mesh gets before export.</summary>
public sealed class ShapeMeshFinisherTests
{
    // A tetrahedron of edge 1 with a repeated face, a face-sized speck far away (area ratio under the small-component threshold) and one
    // face wound the other way.
    private static Mesh Dirty() => new()
    {
        Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 50, 50, 50, 50.0001f, 50, 50, 50, 50.0001f, 50],
        Indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 3, 2, 0, 2, 1, 4, 5, 6],
    };

    [Fact]
    public void Finish_RepairsTheMeshAndGivesItNormals()
    {
        Mesh finished = ShapeMeshFinisher.Finish(Dirty());

        Assert.Equal(4, finished.TriangleCount);
        Assert.Equal(4, finished.VertexCount);
        Assert.True(SyntheticMeshes.IsClosedManifold(finished));
        Assert.NotNull(finished.Normals);
        Assert.Equal(finished.Vertices.Length, finished.Normals!.Length);
    }

    [Fact]
    public void Finish_OrientsAClosedShapeOutward()
    {
        // One face of the dirty tetrahedron is wound against its neighbours; the result agrees and points outward.
        Mesh finished = ShapeMeshFinisher.Finish(Dirty());

        Assert.True(SignedVolume(finished) > 0, "A closed shape comes out wound outward.");
    }

    [Fact]
    public void Finish_ClosesAHoleSmallerThanTheCleanerLimit()
    {
        // A tetrahedron of edge 0.005 missing a face: one boundary loop of perimeter about 0.021, under the default limit of 0.03.
        const float s = 0.005f;
        Mesh open = new()
        {
            Vertices = [0, 0, 0, s, 0, 0, 0, s, 0, 0, 0, s],
            Indices = [0, 2, 1, 0, 1, 3, 0, 3, 2],
        };
        Assert.True(SyntheticMeshes.Topology(open).Boundary > 0);

        Mesh finished = ShapeMeshFinisher.Finish(open);

        Assert.Equal(0, SyntheticMeshes.Topology(finished).Boundary);
    }

    [Fact]
    public void Finish_LeavesALargeOpeningOpen()
    {
        // The unit tetrahedron missing a face: the loop is longer than the hole limit, so boundary edges remain (the documented limitation).
        Mesh open = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            Indices = [0, 2, 1, 0, 1, 3, 0, 3, 2],
        };

        Mesh finished = ShapeMeshFinisher.Finish(open);

        Assert.True(SyntheticMeshes.Topology(finished).Boundary > 0);
        Assert.Equal(3, finished.TriangleCount);
    }

    [Fact]
    public void Finish_WithAnOpeningWidthClosesTheOpeningTheDefaultLeavesOpen()
    {
        // The unit tetrahedron missing a face: a triangle of side 1.4 against a bounding-box diagonal of 1.7.
        Mesh open = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            Indices = [0, 2, 1, 0, 1, 3, 0, 3, 2],
        };

        Mesh closed = ShapeMeshFinisher.Finish(open, openingWidthRatio: 1);
        Mesh stillOpen = ShapeMeshFinisher.Finish(open, openingWidthRatio: ShapeMeshFinisher.DefaultOpeningWidthRatio);

        Assert.True(SyntheticMeshes.IsClosedManifold(closed));
        Assert.True(SignedVolume(closed) > 0);
        Assert.True(SyntheticMeshes.Topology(stillOpen).Boundary > 0);
    }

    [Fact]
    public void Finish_RejectsANegativeOpeningWidth()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ShapeMeshFinisher.Finish(Dirty(), openingWidthRatio: -0.1));
    }

    [Fact]
    public void Finish_WithCleaningOffOnlyUnifiesTheWindingAndKeepsTheFaces()
    {
        Mesh dirty = Dirty();

        Mesh finished = ShapeMeshFinisher.Finish(dirty, clean: false);

        Assert.Same(dirty, finished);
        Assert.Equal(6, finished.TriangleCount);
        Assert.NotNull(finished.Normals);
    }

    [Fact]
    public void Finish_DoesNotModifyTheInputWhenItRepairs()
    {
        Mesh dirty = Dirty();
        int[] before = (int[])dirty.Indices.Clone();

        ShapeMeshFinisher.Finish(dirty);

        Assert.Equal(before, dirty.Indices);
        Assert.Null(dirty.Normals);
    }

    private static double SignedVolume(Mesh mesh)
    {
        double volume = 0;
        for (int f = 0; f < mesh.TriangleCount; f++)
        {
            int a = mesh.Indices[f * 3] * 3, b = mesh.Indices[f * 3 + 1] * 3, c = mesh.Indices[f * 3 + 2] * 3;
            double[] p = [mesh.Vertices[a], mesh.Vertices[a + 1], mesh.Vertices[a + 2]];
            double[] q = [mesh.Vertices[b], mesh.Vertices[b + 1], mesh.Vertices[b + 2]];
            double[] r = [mesh.Vertices[c], mesh.Vertices[c + 1], mesh.Vertices[c + 2]];
            volume += (p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0]) + p[2] * (q[0] * r[1] - q[1] * r[0])) / 6.0;
        }

        return volume;
    }
}
