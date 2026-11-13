using HartsyInference.ThreeD.Geometry;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class GridLimitTests
{
    [Fact]
    public void TheLargestAxisIsTheCubeRootOfOneArrayRoundedDown()
    {
        long largest = GridLimit.LargestAxis, next = largest + 1;

        Assert.True(largest * largest * largest <= Array.MaxLength);
        Assert.True(next * next * next > Array.MaxLength);
        Assert.True(GridLimit.Fits(largest));
        Assert.False(GridLimit.Fits(next));
    }

    [Theory]
    [InlineData(2_097_152L)]
    [InlineData(int.MaxValue)]
    [InlineData(long.MaxValue)]
    [InlineData(-1L)]
    [InlineData(long.MinValue)]
    public void AnAxisWhoseCubeWouldWrapOrIsNegativeDoesNotFit(long axis) => Assert.False(GridLimit.Fits(axis));

    [Fact]
    public void ADefaultGridPassesAndATooLargeOneNamesTheGridParameter()
    {
        GridLimit.RequireDense(0);
        GridLimit.RequireDense(GridLimit.LargestAxis);

        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => GridLimit.RequireDense(GridLimit.LargestAxis + 1));

        Assert.Equal("grid", error.ParamName);
    }
}
