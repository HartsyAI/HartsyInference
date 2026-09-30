using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>The microscaling conventions a block-scaled GEMM operand pair can use. cuBLASLt has to be told which — the scale pointers alone do not say — and the activation quantizer and the checkpoint codecs need the same answer, so the group size, element type and cuBLASLt mode live here once.</summary>
public enum BlockScaleFormat
{
    /// <summary>One E4M3 scale per 16 e2m1 elements — what ComfyUI's <c>nvfp4</c> checkpoints ship.</summary>
    Nvfp4,

    /// <summary>One exponent-only UE8M0 scale per 32 e2m1 elements — OCP microscaling FP4, used by GPT-OSS.</summary>
    Mxfp4,

    /// <summary>One UE8M0 scale per 32 e4m3 elements — OCP microscaling FP8.</summary>
    Mxfp8,
}

public static class BlockScaleFormats
{
    /// <summary>Elements per block scale.</summary>
    public static int GroupSize(this BlockScaleFormat format) => format == BlockScaleFormat.Nvfp4 ? 16 : 32;

    /// <summary>The operand element type cuBLASLt multiplies.</summary>
    public static DType OperandType(this BlockScaleFormat format)
        => format == BlockScaleFormat.Mxfp8 ? DType.F8E4M3 : DType.F4E2M1;

    /// <summary>The <c>cublasLtMatmulMatrixScale_t</c> mode that must accompany the scale pointers.</summary>
    public static int ScaleMode(this BlockScaleFormat format) => format == BlockScaleFormat.Nvfp4
        ? CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
        : CublasLtApi.CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;

    /// <summary>The <see cref="QuantWeightInfo.Format"/> string a checkpoint carries for it.</summary>
    public static string QuantFormat(this BlockScaleFormat format) => format switch
    {
        BlockScaleFormat.Nvfp4 => "nvfp4",
        BlockScaleFormat.Mxfp4 => "mxfp4",
        BlockScaleFormat.Mxfp8 => "mxfp8",
        _ => throw new ArgumentOutOfRangeException(nameof(format), format, null),
    };

    /// <summary>The format behind a <see cref="QuantWeightInfo.Format"/> string, or null for anything that is not block-scaled.</summary>
    public static BlockScaleFormat? FromQuantFormat(string? format) => format switch
    {
        "nvfp4" => BlockScaleFormat.Nvfp4,
        "mxfp4" => BlockScaleFormat.Mxfp4,
        "mxfp8" => BlockScaleFormat.Mxfp8,
        _ => null,
    };
}
