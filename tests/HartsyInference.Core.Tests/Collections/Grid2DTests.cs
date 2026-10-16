using HartsyInference.Core.Collections;
using Xunit;

namespace HartsyInference.Core.Tests.Collections;

/// <summary>A grid survives a trip through its rows, and columns land as the grid's columns.</summary>
public sealed class Grid2DTests
{
    [Fact]
    public void ToRows_ThenFromRows_ReturnsTheSameGrid()
    {
        int[,] grid = { { 1, 2, 3 }, { 4, 5, 6 } };
        int[][] rows = Grid2D.ToRows(grid);
        Assert.Equal([[1, 2, 3], [4, 5, 6]], rows);
        Assert.Equal(grid, Grid2D.FromRows(rows));
    }

    [Fact]
    public void FromColumns_PutsEachColumnDownTheGrid()
    {
        int[,] grid = Grid2D.FromColumns([[1, 4], [2, 5], [3, 6]]);
        Assert.Equal(new[,] { { 1, 2, 3 }, { 4, 5, 6 } }, grid);
    }

    [Fact]
    public void RaggedRowsOrColumns_AreRefused()
    {
        Assert.Throws<ArgumentException>(() => Grid2D.FromRows([[1, 2], [3]]));
        Assert.Throws<ArgumentException>(() => Grid2D.FromColumns([[1, 2], [3]]));
    }

    [Fact]
    public void NoRows_IsAnEmptyGrid()
    {
        int[,] empty = Grid2D.FromRows(Array.Empty<int[]>());
        Assert.Equal((0, 0), (empty.GetLength(0), empty.GetLength(1)));
        Assert.Empty(Grid2D.ToRows(new int[0, 4]));
    }
}
