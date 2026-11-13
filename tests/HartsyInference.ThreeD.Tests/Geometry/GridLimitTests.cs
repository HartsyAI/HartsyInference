using HartsyInference.ThreeD.Geometry;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class GridLimitTests
{
    [Fact]
    public void TheLargestAxisIsTheCubeRootOfOneArray()
    {
        Assert.Equal(1290, GridLimit.LargestAxis());
        Assert.True(GridLimit.Fits(1290));
        Assert.False(GridLimit.Fits(1291));
    }

    [Theory]
    [InlineData(2_097_152L)]
    [InlineData(int.MaxValue)]
    [InlineData(long.MaxValue)]
    public void AnAxisWhoseCubeWouldWrapDoesNotFit(long axis) => Assert.False(GridLimit.Fits(axis));

    [Fact]
    public void ADefaultGridPassesAndATooLargeOneNamesTheGridParameter()
    {
        GridLimit.RequireDense(0);
        GridLimit.RequireDense(1290);

        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => GridLimit.RequireDense(1291));

        Assert.Equal("grid", error.ParamName);
    }
}
