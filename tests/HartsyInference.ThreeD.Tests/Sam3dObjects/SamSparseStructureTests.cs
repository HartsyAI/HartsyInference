using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamSparseStructureTests
{
    private static int[] Cube(int size)
    {
        List<int> coords = [];
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                for (int z = 0; z < size; z++)
                    coords.AddRange([x, y, z]);
            }
        }

        return [.. coords];
    }

    [Fact]
    public void Prune_FilledCubeKeepsOnlyTheShell()
    {
        int[] pruned = SamSparseStructure.Prune(Cube(5));

        // 5³ minus the 3³ interior, and the bounding-box faces count as surface.
        Assert.Equal(125 - 27, pruned.Length / 3);
    }

    [Fact]
    public void Prune_KeepsInputOrderAndRemovesTheCentreVoxel()
    {
        int[] pruned = SamSparseStructure.Prune(Cube(3));

        Assert.Equal(26, pruned.Length / 3);
        Assert.DoesNotContain(true, Enumerable.Range(0, 26).Select(i => pruned[i * 3] == 1 && pruned[i * 3 + 1] == 1 && pruned[i * 3 + 2] == 1));
        Assert.Equal([0, 0, 0, 0, 0, 1], pruned.Take(6));
    }

    [Fact]
    public void Prune_ADistanceOfTwoNeedsAFiveByFiveByFiveNeighbourhood()
    {
        int[] pruned = SamSparseStructure.Prune(Cube(5), 2);

        // Only the centre voxel has a full 5x5x5 neighbourhood.
        Assert.Equal(124, pruned.Length / 3);
    }

    [Fact]
    public void Prune_EmptyAndMisshapenInput()
    {
        Assert.Empty(SamSparseStructure.Prune([]));
        Assert.Throws<ArgumentException>(() => SamSparseStructure.Prune(new int[4]));
    }

    [Fact]
    public void Downsample_AtOrBelowTheCapIsUnchangedWithFactorOne()
    {
        int[] coords = Cube(3);

        (int[] result, int factor) = SamSparseStructure.Downsample(coords, maxCoords: 27);

        Assert.Equal(1, factor);
        Assert.Equal(coords, result);
    }

    [Fact]
    public void Downsample_AboveTheCapHalvesTheGridAboutItsCentre()
    {
        // 4x4x4 over a cap of 10: original 4, target 2, offset 1, so coordinates land in [1, 2] on each axis.
        (int[] result, int factor) = SamSparseStructure.Downsample(Cube(4), maxCoords: 10);

        Assert.Equal(2, factor);
        Assert.Equal(8, result.Length / 3);
        Assert.All(result, v => Assert.InRange(v, 1, 2));
    }

    [Fact]
    public void Downsample_OutputIsSortedAndUnique()
    {
        (int[] result, _) = SamSparseStructure.Downsample(Cube(6), maxCoords: 20);

        List<(int, int, int)> voxels = Enumerable.Range(0, result.Length / 3).Select(i => (result[i * 3], result[i * 3 + 1], result[i * 3 + 2])).ToList();
        Assert.Equal(voxels.Distinct().Count(), voxels.Count);
        Assert.Equal(voxels.OrderBy(v => v).ToList(), voxels);
    }

    [Fact]
    public void Downsample_StillOverTheCapKeepsARandomSubsetOfThatSize()
    {
        (int[] a, int factor) = SamSparseStructure.Downsample(Cube(8), maxCoords: 20, seed: 1);
        (int[] b, _) = SamSparseStructure.Downsample(Cube(8), maxCoords: 20, seed: 1);

        Assert.Equal(2, factor);
        Assert.Equal(20, a.Length / 3);
        Assert.Equal(a, b);
    }

    [Fact]
    public void Downsample_ADegenerateAxisStaysInsideTheGrid()
    {
        int[] flat = [.. Enumerable.Range(0, 30).SelectMany(i => new[] { i, 0, 5 })];

        (int[] result, int factor) = SamSparseStructure.Downsample(flat, maxCoords: 10);

        Assert.Equal(2, factor);
        Assert.NotEmpty(result);
    }
}
