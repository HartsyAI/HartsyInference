using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

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

    /// <summary>Operand types are validated before anything else, so the one pairing cuBLASLt has no kernel for is
    /// named here instead of surfacing as a status code — and the check is exercisable without a card.</summary>
    [Fact]
    public void Run_RefusesTwoE5M2Operands()
    {
        using Fp8GemmExecutor exec = new(8, 6);
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => exec.Run(0, 0, 0, 1, 1, 1, 1.0f, 0, weightType: DType.F8E5M2, inputType: DType.F8E5M2));
        Assert.Contains("E5M2", ex.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(
            () => exec.Run(0, 0, 0, 1, 1, 1, 1.0f, 0, weightType: DType.F16));
    }
}
