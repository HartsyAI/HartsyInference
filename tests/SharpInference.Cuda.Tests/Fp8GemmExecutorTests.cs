using SharpInference.Cuda;
using Xunit;

namespace SharpInference.Cuda.Tests;

public sealed class Fp8GemmExecutorTests
{
    [Theory]
    [InlineData(8, 0, false)]
    [InlineData(8, 6, false)]
    [InlineData(8, 7, false)]
    [InlineData(8, 9, true)]
    [InlineData(9, 0, true)]
    [InlineData(10, 0, true)]
    public void IsSupported_GatesOnAdaPlus(int smMajor, int smMinor, bool expected)
    {
        if (expected)
        {
            return;
        }
        using Fp8GemmExecutor exec = new(smMajor, smMinor);
        Assert.False(exec.IsSupported);
        Assert.Equal(smMajor, exec.SmMajor);
        Assert.Equal(smMinor, exec.SmMinor);
    }

    [Fact]
    public void Run_OnUnsupportedHardware_Throws()
    {
        using Fp8GemmExecutor exec = new(8, 6);
        Assert.False(exec.IsSupported);
        Assert.Throws<InvalidOperationException>(() => exec.Run(0, 0, 0, 1, 1, 1, 1.0f, 0));
    }
}
