using System.Numerics;
using HartsyInference.ThreeD.Geometry;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class DualGridVoxelsTests
{
    [Fact]
    public void ToVoxelOffsets_NonUnitVoxelSize_DividesEachAxisByItsEdgeLength()
    {
        DualGridVoxels voxels = new([2, 1, 0], [1f, 1.5f, 1.25f], [false, false, false]);

        float[] offsets = voxels.ToVoxelOffsets(new Vector3(0.5f, 1f, 2f));

        Assert.Equal([0f, 0.5f, 0.625f], offsets);
    }

    [Fact]
    public void ToVoxelOffsets_LengthMismatch_ThrowsInvalidOperation()
    {
        DualGridVoxels voxels = new([0, 0, 0], [0f, 0f], [false, false, false]);

        Assert.Throws<InvalidOperationException>(() => voxels.ToVoxelOffsets(Vector3.One));
    }

    [Fact]
    public void ToVoxelOffsets_CoordsNotMultipleOfThree_ThrowsInvalidOperation()
    {
        DualGridVoxels voxels = new([0, 0], [0f, 0f], [false, false]);

        Assert.Throws<InvalidOperationException>(() => voxels.ToVoxelOffsets(Vector3.One));
    }

    [Fact]
    public void ToVoxelOffsets_NonPositiveVoxelSize_Throws()
    {
        DualGridVoxels voxels = new([0, 0, 0], [0f, 0f, 0f], [false, false, false]);

        Assert.Throws<ArgumentException>(() => voxels.ToVoxelOffsets(new Vector3(1f, 0f, 1f)));
    }
}
