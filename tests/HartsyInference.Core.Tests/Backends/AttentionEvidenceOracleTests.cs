using HartsyInference.GpuBenchmarks.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The evidence oracle must distinguish exact excluded-row zeros from errors and nonfinite outputs.</summary>
public sealed class AttentionEvidenceOracleTests
{
    [Fact]
    public void BottomRightAlignmentWithMoreQueriesThanKeysHasEmptyLeadingRows()
    {
        AttentionBenchShape shape = new("empty", 1, 1, 1, 3, 1, 2, "hm", "none", "flash", true, "");
        AttentionOracle oracle = new(shape, new float[6], new float[2], [2, 4], null, 1);
        Assert.Equal(new double[2], oracle.Row(0, 0, 0));
        Assert.Equal(new double[2], oracle.Row(0, 0, 1));
        Assert.Equal(new double[] { 2, 4 }, oracle.Row(0, 0, 2));
    }

    [Fact]
    public void ZeroReferenceAcceptsOnlyExactFiniteZero()
    {
        AttentionBenchShape shape = new("zero", 1, 1, 1, 1, 1, 2, "hm", "none", "sdpa", false, "");
        AttentionOracle oracle = new(shape, new float[2], new float[2], new float[2], null, 1);
        Assert.Equal(0, oracle.Compare(static (_, _, _, _) => 0, 1).RelL2);
        Assert.Equal(double.PositiveInfinity, oracle.Compare(static (_, _, _, _) => 1, 1).RelL2);
        Assert.Equal(double.PositiveInfinity, oracle.Compare(static (_, _, _, _) => double.NaN, 1).RelL2);
    }
}
