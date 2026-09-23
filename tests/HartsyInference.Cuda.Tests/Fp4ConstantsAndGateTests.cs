using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins the FP4 interop constants and the Blackwell gate.
///
/// <para>These are transcribed from <c>library_types.h</c> and <c>cublasLt.h</c>, and a wrong value is invisible
/// locally: no card here runs the FP4 path, so nothing would fail until it reached Blackwell and either errored
/// obscurely or multiplied by a misread scale. <c>CUDA_R_8F_UE8M0</c> was 34 — past the end of <c>cudaDataType</c>
/// — for exactly that reason.</para>
///
/// <para>Unit tier: no GPU, no CUDA runtime. Values below are the enum members, not a second opinion about them;
/// re-derive from the headers rather than from this file if they ever disagree.</para></summary>
public sealed class Fp4ConstantsAndGateTests
{
    [Theory]
    // cudaDataType, library_types.h
    [InlineData("CUDA_R_4F_E2M1", 33)]
    [InlineData("CUDA_R_6F_E2M3", 31)]
    [InlineData("CUDA_R_6F_E3M2", 32)]
    [InlineData("CUDA_R_8F_UE4M3", 28)]   // alias of CUDA_R_8F_E4M3
    [InlineData("CUDA_R_8F_UE8M0", 30)]
    public void CudaDataTypeConstantsMatchTheHeader(string name, int expected)
    {
        int actual = name switch
        {
            "CUDA_R_4F_E2M1" => CublasApi.CUDA_R_4F_E2M1,
            "CUDA_R_6F_E2M3" => CublasApi.CUDA_R_6F_E2M3,
            "CUDA_R_6F_E3M2" => CublasApi.CUDA_R_6F_E3M2,
            "CUDA_R_8F_UE4M3" => CublasApi.CUDA_R_8F_UE4M3,
            "CUDA_R_8F_UE8M0" => CublasApi.CUDA_R_8F_UE8M0,
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unmapped constant"),
        };
        Assert.Equal(expected, actual);
    }

    [Theory]
    // cublasLtMatmulDescAttributes_t and cublasLtMatmulMatrixScale_t, cublasLt.h
    [InlineData("A_SCALE_MODE", 31)]
    [InlineData("B_SCALE_MODE", 32)]
    [InlineData("VEC16_UE4M3", 1)]
    [InlineData("VEC32_UE8M0", 2)]
    public void BlockScalingConstantsMatchTheHeader(string name, int expected)
    {
        int actual = name switch
        {
            "A_SCALE_MODE" => CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            "B_SCALE_MODE" => CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            "VEC16_UE4M3" => CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
            "VEC32_UE8M0" => CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unmapped constant"),
        };
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(8, 6, false)]    // Ampere — this box's 3060
    [InlineData(8, 9, false)]    // Ada — this box's 4090
    [InlineData(9, 0, false)]    // Hopper: FP8 tensor cores, no FP4
    [InlineData(10, 0, true)]    // Blackwell datacenter
    [InlineData(12, 0, true)]    // Blackwell consumer (RTX 50xx)
    public void IsSupportedFollowsComputeCapability(int major, int minor, bool expected)
    {
        // Constructing an unsupported executor must allocate no native resources, which is what lets callers ask
        // unconditionally. The supported branch would create a cuBLASLt handle, so it is not exercised here.
        if (expected)
        {
            return;
        }
        using Fp4GemmExecutor executor = new(major, minor);
        Assert.False(executor.IsSupported);
    }

    [Fact]
    public void RunRefusesOnNonBlackwellRatherThanProducingGarbage()
    {
        using Fp4GemmExecutor executor = new(8, 9);
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => executor.Run(0, 0, 0, 0, 0, m: 1, n: 1, k: 16, stream: 0));
        Assert.Contains("Blackwell", ex.Message, StringComparison.Ordinal);
    }
}
