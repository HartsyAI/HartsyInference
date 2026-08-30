using HartsyInference.Core.Numerics;
using Xunit;

namespace HartsyInference.Core.Tests;

public sealed class UnitIntervalTests
{
    [Fact]
    public void ContainsAcceptsOnlyFiniteInclusiveNormalizedValues()
    {
        Assert.True(UnitInterval.Contains(0f));
        Assert.True(UnitInterval.Contains(0.5f));
        Assert.True(UnitInterval.Contains(1f));
        Assert.False(UnitInterval.Contains(-float.Epsilon));
        Assert.False(UnitInterval.Contains(MathF.BitIncrement(1f)));
        Assert.False(UnitInterval.Contains(float.NaN));
        Assert.False(UnitInterval.Contains(float.PositiveInfinity));
    }
}
