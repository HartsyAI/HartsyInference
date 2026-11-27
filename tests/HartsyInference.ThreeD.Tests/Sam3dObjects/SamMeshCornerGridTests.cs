using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamMeshCornerGridTests
{
    [Fact]
    public void Build_ListsTheDistinctCornersSortedAndAveragesWhatSharingCubesSay()
    {
        // Two cubes that share the face x = 1: (0, 0, 0) and (1, 0, 0). One channel: cube 0 says 10 * corner index, cube 1 says 100 + corner index.
        int[] cubes = [0, 0, 0, 1, 0, 0];
        float[] values = new float[2 * 8];
        for (int j = 0; j < 8; j++)
        {
            values[j] = 10 * j;
            values[8 + j] = 100 + j;
        }

        SamMeshCornerGrid grid = SamMeshCornerGrid.Build(cubes, values, 1);

        Assert.Equal(12, grid.PointCount);
        // Sorted by (x, y, z): the first point is (0, 0, 0), the last (2, 1, 1).
        Assert.Equal([0, 0, 0], grid.Points[0..3]);
        Assert.Equal([2, 1, 1], grid.Points[33..36]);
        for (int p = 1; p < grid.PointCount; p++)
        {
            (int, int, int) a = (grid.Points[p * 3 - 3], grid.Points[p * 3 - 2], grid.Points[p * 3 - 1]);
            (int, int, int) b = (grid.Points[p * 3], grid.Points[p * 3 + 1], grid.Points[p * 3 + 2]);
            Assert.True(a.CompareTo(b) < 0);
        }

        // Cube 0's corner 1 is (1, 0, 0), cube 1's corner 0: the same point, value (10 + 100) / 2.
        int shared = grid.CubeCorners[1];
        Assert.Equal(grid.CubeCorners[8], shared);
        Assert.Equal(55f, grid.Attributes[shared], 4);
        // Cube 0's corner 0 (0, 0, 0) belongs to that cube only.
        Assert.Equal(0f, grid.Attributes[grid.CubeCorners[0]], 4);
        // Cube 1's corner 7 (2, 1, 1) only.
        Assert.Equal(107f, grid.Attributes[grid.CubeCorners[15]], 4);
    }

    [Fact]
    public void Build_RejectsValuesThatDoNotMatchTheCubes()
    {
        Assert.Throws<ArgumentException>(() => SamMeshCornerGrid.Build([0, 0, 0], new float[7], 1));
        Assert.Throws<ArgumentException>(() => SamMeshCornerGrid.Build([0, 0], new float[8], 1));
    }
}
