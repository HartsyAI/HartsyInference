using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins the block-scaled GEMM interop constants, the dtype map, the format descriptor and the Blackwell gate.
///
/// <para>These are transcribed from <c>library_types.h</c> and <c>cublasLt.h</c>, and a wrong value is invisible
/// locally: no card here runs the block-scaled path, so nothing would fail until it reached Blackwell and either
/// errored obscurely or multiplied by a misread scale. <c>CUDA_R_8F_UE8M0</c> was 34 — past the end of
/// <c>cudaDataType</c> — for exactly that reason.</para>
///
/// <para>Unit tier: no GPU, no CUDA runtime. Values below are the enum members, not a second opinion about them;
/// re-derive from the headers rather than from this file if they ever disagree.</para></summary>
public sealed class BlockScaledConstantsAndGateTests
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

    /// <summary>Every layout an executor creates goes through this map, so it is the one place an operand's element
    /// type can be wrong. The E5M2 row is the one that used to be missing: the fp8 executor spelled E4M3 for both
    /// operands and would have multiplied an E5M2 weight as E4M3.</summary>
    [Theory]
    [InlineData("F32", 0)]
    [InlineData("F16", 2)]
    [InlineData("BF16", 14)]
    [InlineData("F8_E4M3", 28)]
    [InlineData("F8_E5M2", 29)]
    [InlineData("F4_E2M1", 33)]
    [InlineData("I8", 3)]
    [InlineData("I32", 10)]
    public void DataTypeOfMatchesTheHeader(string dtypeName, int expected)
    {
        DType dtype = dtypeName switch
        {
            "F32" => DType.F32, "F16" => DType.F16, "BF16" => DType.BF16,
            "F8_E4M3" => DType.F8E4M3, "F8_E5M2" => DType.F8E5M2, "F4_E2M1" => DType.F4E2M1,
            "I8" => DType.I8, "I32" => DType.I32,
            _ => throw new ArgumentOutOfRangeException(nameof(dtypeName), dtypeName, "unmapped dtype"),
        };
        Assert.Equal(expected, CublasApi.DataTypeOf(dtype));
    }

    [Fact]
    public void DataTypeOfRefusesWhatCublasCannotHold()
    {
        Assert.Throws<NotSupportedException>(() => CublasApi.DataTypeOf(DType.Q4_K));
    }

    [Theory]
    // cublasLtMatmulDescAttributes_t, cublasLtMatmulMatrixScale_t and cublasLtPointerMode_t, cublasLt.h
    [InlineData("POINTER_MODE", 2)]
    [InlineData("POINTER_MODE_DEVICE", 1)]
    [InlineData("A_SCALE_MODE", 31)]
    [InlineData("B_SCALE_MODE", 32)]
    [InlineData("VEC16_UE4M3", 1)]
    [InlineData("VEC32_UE8M0", 2)]
    public void BlockScalingConstantsMatchTheHeader(string name, int expected)
    {
        int actual = name switch
        {
            "POINTER_MODE" => CublasLtApi.CUBLASLT_MATMUL_DESC_POINTER_MODE,
            "POINTER_MODE_DEVICE" => CublasLtApi.CUBLASLT_POINTER_MODE_DEVICE,
            "A_SCALE_MODE" => CublasLtApi.CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            "B_SCALE_MODE" => CublasLtApi.CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            "VEC16_UE4M3" => CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
            "VEC32_UE8M0" => CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unmapped constant"),
        };
        Assert.Equal(expected, actual);
    }

    /// <summary>One descriptor per format: the quantizer, the executor and the checkpoint codecs all read it, so its
    /// rows are what "NVFP4" means everywhere.</summary>
    [Theory]
    [InlineData(BlockScaleFormat.Nvfp4, 16, "F4_E2M1", 1, "nvfp4")]
    [InlineData(BlockScaleFormat.Mxfp4, 32, "F4_E2M1", 2, "mxfp4")]
    [InlineData(BlockScaleFormat.Mxfp8, 32, "F8_E4M3", 2, "mxfp8")]
    public void FormatDescriptorRows(BlockScaleFormat format, int group, string operand, int mode, string quant)
    {
        Assert.Equal(group, format.GroupSize());
        Assert.Equal(operand, format.OperandType().Name);
        Assert.Equal(mode, format.ScaleMode());
        Assert.Equal(quant, format.QuantFormat());
        Assert.Equal(format, BlockScaleFormats.FromQuantFormat(quant));
    }

    [Fact]
    public void OnlyBlockScaledFormatsResolve()
    {
        Assert.Null(BlockScaleFormats.FromQuantFormat("int8_tensorwise"));
        Assert.Null(BlockScaleFormats.FromQuantFormat(null));
    }

    /// <summary>The one arch predicate every gate reads. SM 10.x sits between the "major >= 12" the cuBLAS warning
    /// used to test and the "major >= 10" the FP4 executor tested, which is how one Blackwell tier could pass a gate
    /// the other failed.</summary>
    [Theory]
    [InlineData(8, 6, false, false)]   // Ampere — this box's 3060
    [InlineData(8, 9, true, false)]    // Ada — this box's 4090
    [InlineData(9, 0, true, false)]    // Hopper: FP8, no FP4
    [InlineData(10, 0, true, true)]    // Blackwell datacenter
    [InlineData(10, 3, true, true)]    // B300
    [InlineData(12, 0, true, true)]    // Blackwell consumer (RTX 50xx)
    public void ArchTiersOrderTheWayTheHardwareDoes(int major, int minor, bool fp8, bool fp4)
    {
        int sm = CudaArch.Sm(major, minor);
        Assert.Equal(fp8, sm >= CudaArch.Ada);
        Assert.Equal(fp4, sm >= CudaArch.Blackwell);
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
        using BlockScaledGemmExecutor executor = new(major, minor);
        Assert.False(executor.IsSupported);
    }

    [Fact]
    public void RunRefusesOnNonBlackwellRatherThanProducingGarbage()
    {
        using BlockScaledGemmExecutor executor = new(8, 9);
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => executor.Run(0, 0, 0, 0, 0, m: 1, n: 1, k: 16, alphaBetaDev: 0, stream: 0, format: BlockScaleFormat.Nvfp4));
        Assert.Contains("Blackwell", ex.Message, StringComparison.Ordinal);
    }
}
