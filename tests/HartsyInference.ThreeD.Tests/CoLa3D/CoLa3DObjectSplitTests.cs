using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

public sealed class CoLa3DObjectSplitTests
{
    [Fact]
    public void AssignOwners_TakesTheHighestPositiveLogitAndDropsNonPositiveRows()
    {
        float[] logits = [1f, 2f, -1f, -2f, 0.5f, -0.1f];

        int[] owners = CoLa3DObjectSplit.AssignOwners(logits, voxels: 3, parts: 2);

        Assert.Equal([1, -1, 0], owners);
    }

    [Fact]
    public void AssignOwners_RejectsLogitsOfTheWrongSize()
    {
        float[] logits = [1f, 2f, 3f];

        Assert.Throws<ArgumentException>(() => CoLa3DObjectSplit.AssignOwners(logits, voxels: 2, parts: 2));
    }

    [Fact]
    public void GroupRows_KeepsOnlyPartsWithEnoughVoxelsInVoxelOrder()
    {
        int[] owners = new int[20];
        for (int i = 0; i < 20; i++)
            owners[i] = i < 10 ? 0 : (i < 15 ? 1 : -1);

        List<int[]> groups = CoLa3DObjectSplit.GroupRows(owners, parts: 2);

        Assert.Single(groups);
        Assert.Equal(Enumerable.Range(0, 10).ToArray(), groups[0]);
    }
}
