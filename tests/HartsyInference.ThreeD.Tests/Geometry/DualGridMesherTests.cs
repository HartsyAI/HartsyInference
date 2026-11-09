using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class DualGridMesherTests
{
    [Fact]
    public void SingleVoxel_EmitsNoQuadsAndScalesVertexIntoWorldSpace()
    {
        DualGridMesh mesh = DualGridMesher.Extract(
            coords: [0, 0, 0],
            dualVertices: [0.5f, 0.5f, 0.5f],
            intersected: [true, true, true],
            splitWeight: null,
            aabbMin: new Vector3(-1f, -1f, -1f),
            voxelSize: Vector3.One);

        Assert.Empty(mesh.Triangles);
        Assert.Equal(new[] { -0.5f, -0.5f, -0.5f }, mesh.Vertices);
    }

    [Fact]
    public void TwoByTwoByTwoBlock_EmitsSixQuadsAsTwelveTrianglesOverEightVertices()
    {
        int[] coords = Corners();
        float[] dual = Enumerable.Repeat(0.5f, 24).ToArray();
        bool[] flags = Enumerable.Repeat(true, 24).ToArray();

        DualGridMesh mesh = DualGridMesher.Extract(coords, dual, flags, null, Vector3.Zero, Vector3.One);

        Assert.Equal(8 * 3, mesh.Vertices.Length);
        Assert.Equal(12 * 3, mesh.Triangles.Length);
        Assert.All(mesh.Triangles, t => Assert.InRange(t, 0, 7));
    }

    [Fact]
    public void Quad_SplitFollowsWeightProductComparison()
    {
        // Four voxels in the x=0 plane form one x-axis quad from voxel 0; the others emit nothing.
        int[] coords = { 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0 };
        float[] dual = Enumerable.Repeat(0.5f, 12).ToArray();
        bool[] flags = { true, false, false, false, false, false, false, false, false, false, false, false };

        DualGridMesh equal = DualGridMesher.Extract(coords, dual, flags, new float[] { 1f, 1f, 1f, 1f }, Vector3.Zero, Vector3.One);
        // w02 = 2*2 = 4 > w13 = 1*1 = 1: split 1.
        DualGridMesh split1 = DualGridMesher.Extract(coords, dual, flags, new float[] { 2f, 1f, 2f, 1f }, Vector3.Zero, Vector3.One);

        Assert.Equal(new[] { 0, 1, 3, 3, 1, 2 }, equal.Triangles);
        Assert.Equal(new[] { 0, 1, 2, 0, 2, 3 }, split1.Triangles);
    }

    [Fact]
    public void Quad_WithoutWeights_UsesReferenceAlignmentTest_SelectingSplitOne()
    {
        // Documents the reference quirk: the split-1 alignment compares a triangle with itself and the split-2 test
        // gives a zero normal, so split 1 is selected for a non-degenerate quad.
        int[] coords = { 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0 };
        float[] dual = Enumerable.Repeat(0.5f, 12).ToArray();
        bool[] flags = { true, false, false, false, false, false, false, false, false, false, false, false };

        DualGridMesh mesh = DualGridMesher.Extract(coords, dual, flags, null, Vector3.Zero, Vector3.One);

        Assert.Equal(new[] { 0, 1, 2, 0, 2, 3 }, mesh.Triangles);
    }

    [Fact]
    public void DuplicateVoxelCoordinate_IsRejected()
    {
        Assert.Throws<ArgumentException>(() => DualGridMesher.Extract(
            new[] { 0, 0, 0, 0, 0, 0 },
            new float[6], new bool[6], null, Vector3.Zero, Vector3.One));
    }

    private static int[] Corners() => new[]
    {
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
    };
}
